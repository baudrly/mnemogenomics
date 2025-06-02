#!/usr/bin/env python3
import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT
from fasta_utils import FastaTokenizer # For loading meta.pkl during resume

# -----------------------------------------------------------------------------
# default config values designed to train a model on FASTA data
# I/O
out_dir = 'out-fasta'
eval_interval = 1000 # Lowered for potentially smaller datasets
log_interval = 10 # Log more frequently
eval_iters = 100 # Fewer eval iterations
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'fasta-gpt'
wandb_run_name = 'run' + str(int(time.time()))
# data
dataset = 'fasta_prepared' # This should match the output dir of prepare_fasta_data.py
gradient_accumulation_steps = 4 # Adjust based on GPU memory and desired batch size
batch_size = 8 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 256 # Max sequence length for training, should match prepare_fasta_data.py
# model
n_layer = 6
n_head = 6
n_embd = 384 # Embedding dimension
dropout = 0.1 # Increased dropout for smaller models/datasets
bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 3e-4 # max learning rate
max_iters = 50000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 1000 # how many steps to warm up for
lr_decay_iters = 50000 # should be ~= max_iters
min_lr = 3e-5 # minimum learning rate, should be ~= learning_rate/10
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
if master_process: print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# poor man's data loader for .bin files
data_dir = os.path.join('data', dataset)
train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
tokenizer_meta = None # To store tokenizer related metadata if needed later
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = len(meta['vocab']) # FastaTokenizer saves vocab list
    tokenizer_meta = meta # Store all meta for potential use (e.g. eos_id in generation)
    if master_process: print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")


# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line
if init_from == 'scratch':
    if master_process: print("Initializing a new model from scratch")
    if meta_vocab_size is None:
        raise ValueError("meta.pkl not found or vocab_size not in meta.pkl. Run prepare_fasta_data.py first.")
    model_args['vocab_size'] = meta_vocab_size
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    if master_process: print(f"Resuming training from {out_dir}")
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    if master_process: print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
    # IMPORTANT: if using pretrained GPT-2, the vocab size will be different.
    # This is generally not recommended for FASTA task unless you specifically map your FASTA vocab to GPT-2's.
    # For this script, it's assumed `init_from=scratch` or `resume` will use the FASTA vocab.
    if meta_vocab_size is not None and model_args['vocab_size'] != meta_vocab_size:
        print(f"Warning: GPT-2 vocab size ({model_args['vocab_size']}) != FASTA vocab size ({meta_vocab_size}).")
        print("This might lead to issues unless vocabulary is properly handled (e.g. re-embedding layer).")


if block_size < model.config.block_size: # crop down model block size if needed
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size 
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume' and 'optimizer' in checkpoint: # check for optimizer state
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model
if compile:
    if master_process: print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    try:
        model = torch.compile(model) # requires PyTorch 2.0
    except Exception as e:
        if master_process: print(f"Model compilation failed: {e}. Proceeding without compilation.")
        compile = False # Disable compile if it fails

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def get_lr(it):
    if not decay_lr: return learning_rate
    if it < warmup_iters: # linear warmup
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters: # constant min_lr after decay
        return min_lr
    # cosine decay
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 
raw_model = model.module if ddp else model 
running_mfu = -1.0
while True:
    lr = get_lr(iter_num)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        val_loss = losses['val']
        train_loss = losses['train']
        perplexity_val = math.exp(val_loss) if val_loss < 20 else float('inf') # cap perplexity for display
        perplexity_train = math.exp(train_loss) if train_loss < 20 else float('inf')
        
        print(f"step {iter_num}: train loss {train_loss:.4f} (ppl {perplexity_train:.2f}), val loss {val_loss:.4f} (ppl {perplexity_val:.2f})")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": train_loss,
                "train/perplexity": perplexity_train,
                "val/loss": val_loss,
                "val/perplexity": perplexity_val,
                "lr": lr,
                "mfu": running_mfu*100, # convert to percentage
            })
        if val_loss < best_val_loss or always_save_checkpoint:
            best_val_loss = val_loss
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                if tokenizer_meta: # Save tokenizer metadata if available
                    checkpoint['tokenizer_meta'] = tokenizer_meta
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
                # Save a separate .pth for just the model weights for easier loading elsewhere
                torch.save(raw_model.state_dict(), os.path.join(out_dir, 'model_weights.pth'))


    if iter_num == 0 and eval_only:
        break

    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps 
        X, Y = get_batch('train')
        scaler.scale(loss).backward()
    
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%, lr {current_lr:.2e}")
    iter_num += 1
    local_iter_num += 1

    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()