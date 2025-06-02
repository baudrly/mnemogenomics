#!/usr/bin/env python3
import os
import pickle
from contextlib import nullcontext
import torch
from model import GPTConfig, GPT
from fasta_utils import FastaTokenizer, detect_sequence_type, clean_header

# -----------------------------------------------------------------------------
# Configuration for sampling (can be overridden by command line args via configurator.py)
init_from = 'resume' # 'resume' (from out_dir) or 'gpt2*' (pre-trained GPT-2 model)
out_dir = 'out-fasta' # Directory of the trained model, ignored if init_from is not 'resume'
start_mode = 'sequence' # 'auto', 'description', or 'sequence'. 'auto' tries to guess.
start_input = "" # Prompt string (description or sequence). Can also be "FILE:prompt.txt"
num_samples = 1 # Number of samples to generate
max_new_tokens = 200 # Maximum number of new tokens to generate per sample
temperature = 0.8 # Sampling temperature (1.0 = no change, <1.0 = less random, >1.0 = more random)
top_k = 50 # Retain only top_k most likely tokens (<=0 to disable)
seed = 1337
device = 'cuda' # 'cpu', 'cuda', 'cuda:0', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = False # Use PyTorch 2.0 to compile the model (faster inference)
# -----------------------------------------------------------------------------
# Attempt to load configurator.py for command line overrides
try:
    exec(open('configurator.py').read())
except FileNotFoundError:
    print("configurator.py not found, using default CLI args.")
except Exception as e:
    print(f"Error executing configurator.py: {e}")
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed) # if using CUDA
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# Load model and tokenizer
if init_from == 'resume':
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    
    # Load tokenizer metadata
    meta_path = os.path.join(out_dir, 'meta.pkl')
    if not os.path.exists(meta_path) and 'tokenizer_meta' in checkpoint: # Fallback to checkpoint's tokenizer_meta
        print("Loading tokenizer_meta from checkpoint.")
        tokenizer = FastaTokenizer(vocab=checkpoint['tokenizer_meta']['vocab'])
        for key, value in checkpoint['tokenizer_meta'].items():
            if key not in ['vocab', 'stoi', 'itos']:
                 setattr(tokenizer, key, value)
    elif os.path.exists(meta_path):
        print(f"Loading tokenizer metadata from {meta_path}")
        tokenizer = FastaTokenizer.from_file(meta_path)
    else:
        raise FileNotFoundError(f"meta.pkl not found in {out_dir} and tokenizer_meta not in checkpoint. Cannot initialize tokenizer.")

elif init_from.startswith('gpt2'):
    # This path is less likely for FASTA, as vocabularies differ significantly.
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))
    print("Using generic TikToken GPT-2 tokenizer for GPT-2 pretrained model.")
    print("WARNING: This is likely unsuitable for FASTA specific tasks unless model was fine-tuned extensively.")
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    # Create a mock FastaTokenizer interface
    class MockFastaTokenizer:
        def __init__(self, enc_tiktoken):
            self.enc = enc_tiktoken
            self.bos_id = enc_tiktoken.eot_token # Use EOT as BOS for GPT-2
            self.eos_id = enc_tiktoken.eot_token
            self.sep_id = enc_tiktoken.encode("\n\n")[0] # Arbitrary separator
            self.dna_marker_id = enc_tiktoken.encode("[DNA]")[0] # Will be single tokens if not in vocab
            self.rna_marker_id = enc_tiktoken.encode("[RNA]")[0]
            self.protein_marker_id = enc_tiktoken.encode("[PROTEIN]")[0]
            self.desc_marker_id = enc_tiktoken.encode("[DESC]")[0]
            self.vocab_size = enc_tiktoken.n_vocab
        def encode(self, s, is_sequence_data=False): # Ignore is_sequence_data for tiktoken
            return self.enc.encode(s)
        def decode(self, ids):
            return self.enc.decode(ids)
    tokenizer = MockFastaTokenizer(enc)
else:
    raise ValueError(f"Unknown init_from type: {init_from}")

model.eval()
model.to(device)
if compile:
    try:
        model = torch.compile(model)
    except Exception as e:
        print(f"Model compilation failed: {e}. Proceeding without compilation.")


# --- Interactive CLI or File Input ---
if not start_input:
    start_input = input("Enter starting description or sequence (or 'FILE:path/to/file.txt'):\n> ")

if start_input.startswith('FILE:'):
    with open(start_input[5:], 'r', encoding='utf-8') as f:
        start_input = f.read().strip()

# Determine input type (description or sequence)
current_mode = start_mode
if current_mode == 'auto':
    # Simple heuristic: if it contains DNA/RNA/Protein specific characters and is longish, assume sequence
    # Otherwise, assume description.
    _seq_type_guess = detect_sequence_type(start_input)
    if _seq_type_guess != "UNKNOWN" and len(start_input) > 20 and \
       any(c.upper() in "ACGTUN" + "ARNDCQEGHILKMFPSTWYVBZXJO" for c in start_input):
        current_mode = 'sequence'
        print("Auto-detected input as SEQUENCE.")
    else:
        current_mode = 'description'
        print("Auto-detected input as DESCRIPTION.")
elif current_mode not in ['description', 'sequence']:
    print(f"Invalid start_mode '{current_mode}'. Defaulting to 'description'.")
    current_mode = 'description'


# Prepare the prompt
prompt_tokens = [tokenizer.bos_id]

if current_mode == 'description':
    print(f"Mode: DESCRIPTION -> SEQUENCE")
    cleaned_description = clean_header(start_input) # Clean even if user provides it
    tokenized_description = tokenizer.encode(cleaned_description, is_sequence_data=False)
    prompt_tokens.extend([tokenizer.desc_marker_id] + tokenized_description + [tokenizer.sep_id])
    
    # Ask user for target sequence type or provide options
    target_seq_type_str = input("Enter target sequence type (DNA, RNA, Protein): ").upper()
    if target_seq_type_str == "DNA":
        prompt_tokens.append(tokenizer.dna_marker_id)
    elif target_seq_type_str == "RNA":
        prompt_tokens.append(tokenizer.rna_marker_id)
    elif target_seq_type_str == "PROTEIN":
        prompt_tokens.append(tokenizer.protein_marker_id)
    else:
        print("Invalid sequence type. Defaulting to DNA.")
        prompt_tokens.append(tokenizer.dna_marker_id)
    
    print(f"Generating sequence for description: '{cleaned_description}' (Type: {target_seq_type_str})")

elif current_mode == 'sequence':
    print(f"Mode: SEQUENCE -> DESCRIPTION")
    seq_type = detect_sequence_type(start_input)
    if seq_type == "DNA":
        prompt_tokens.append(tokenizer.dna_marker_id)
    elif seq_type == "RNA":
        prompt_tokens.append(tokenizer.rna_marker_id)
    elif seq_type == "PROTEIN":
        prompt_tokens.append(tokenizer.protein_marker_id)
    else: # Unknown sequence type
        print("Could not determine sequence type. Please add type markers like [DNA], [RNA], [PROTEIN] or ensure sequence is standard.")
        exit(1)
    
    tokenized_sequence = tokenizer.encode(start_input, is_sequence_data=True)
    prompt_tokens.extend(tokenized_sequence + [tokenizer.sep_id, tokenizer.desc_marker_id])
    print(f"Generating description for sequence (type {seq_type}): '{start_input[:100]}...'")

# Convert prompt to tensor
x = (torch.tensor(prompt_tokens, dtype=torch.long, device=device)[None, ...])

# Run generation
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            print(f"\n--- Sample {k+1}/{num_samples} ---")
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k if top_k > 0 else None, eos_token_id=tokenizer.eos_id)
            generated_text = tokenizer.decode(y[0].tolist())
            
            # Post-process to make output cleaner
            # Remove BOS token if present at start, and anything after EOS
            if generated_text.startswith(tokenizer.decode([tokenizer.bos_id])):
                generated_text = generated_text[len(tokenizer.decode([tokenizer.bos_id])):]
            if tokenizer.decode([tokenizer.eos_id]) in generated_text:
                generated_text = generated_text.split(tokenizer.decode([tokenizer.eos_id]))[0]

            # Print interpretation based on mode
            parts = generated_text.split(tokenizer.decode([tokenizer.sep_id]))
            
            # Print the full raw output first
            print("Raw generated output:\n", tokenizer.decode(y[0].tolist())) # Raw, includes prompt
            print("\nProcessed output:")
            
            if len(parts) >= 2:
                # Assuming the input prompt part is the first element before SEP implicitly
                # The generated part is after the SEP
                generated_content_full = parts[-1] # Take the last part after all SEPs
                
                # Try to remove type/desc markers from the start of the generated content
                type_marker_tokens = [tokenizer.decode([m]) for m in [tokenizer.dna_marker_id, tokenizer.rna_marker_id, tokenizer.protein_marker_id, tokenizer.desc_marker_id]]
                for marker_str in type_marker_tokens:
                    if generated_content_full.startswith(marker_str):
                        generated_content_full = generated_content_full[len(marker_str):]
                        break
                
                if current_mode == 'description': # Generated a sequence
                    print(f"Generated Sequence: {generated_content_full.strip().upper()}")
                elif current_mode == 'sequence': # Generated a description
                    print(f"Generated Description: {generated_content_full.strip().lower()}")
            else:
                # If SEP was not generated or format is unexpected, print relevant part
                # This usually means generation stopped early or produced unexpected format
                # We take text after the prompt
                prompt_decoded_len = len(tokenizer.decode(prompt_tokens))
                # The y[0] is the full sequence (prompt + generation)
                # We need to decode only the generated part from y[0]
                generated_ids_only = y[0][len(prompt_tokens):].tolist()
                generated_text_only = tokenizer.decode(generated_ids_only)
                if tokenizer.decode([tokenizer.eos_id]) in generated_text_only:
                     generated_text_only = generated_text_only.split(tokenizer.decode([tokenizer.eos_id]))[0]

                print(f"Generated content (partial/direct): {generated_text_only.strip()}")

            print('---------------')