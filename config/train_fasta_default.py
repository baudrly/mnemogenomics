# Default configuration for training on prepared FASTA data
# This can be used as a base and overridden via command line or another config file.

# I/O
out_dir = 'out-fasta-model'
eval_interval = 500  # Evaluate more often on potentially smaller/faster training
log_interval = 20    # Log training progress
eval_iters = 100
eval_only = False 
always_save_checkpoint = True 
init_from = 'scratch' # 'scratch' or 'resume'

# wandb logging
wandb_log = False # Set to True to use wandb
wandb_project = 'fasta-gpt-proj'
wandb_run_name = 'fasta-run' # Will be 'run' + timestamp if not overridden

# data
dataset = 'fasta_prepared' # This should be the directory created by prepare_fasta_data.py
                           # e.g., if output of prepare_fasta_data.py is 'data/my_fasta_data', then dataset = 'my_fasta_data'
gradient_accumulation_steps = 8 
batch_size = 16                 # Micro-batch size
block_size = 512               # Context window size, should match prepare_fasta_data.py's max_seq_len

# model - A smaller model might be suitable for character-level tasks or smaller datasets
n_layer = 8
n_head = 8
n_embd = 512 # Embedding dimension (e.g., 384, 512, 768)
dropout = 0.1 
bias = False # Using bias=False can sometimes improve performance slightly

# AdamW optimizer
learning_rate = 5e-4 # Can be higher for smaller models
max_iters = 100000   # Total training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95         # Slightly lower beta2 can sometimes help
grad_clip = 1.0      # Gradient clipping

# Learning rate decay settings
decay_lr = True
warmup_iters = 2000  # Warmup iterations
lr_decay_iters = 100000 # Should generally match max_iters
min_lr = 5e-5        # Minimum learning rate

# System settings
device = 'cuda' # 'cuda' or 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = True      # Use PyTorch 2.0 compilation if available

# DDP settings (if using distributed training)
backend = 'nccl' 