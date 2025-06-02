import os
import torch
import pickle
import json
import subprocess
from model import GPT, GPTConfig
from fasta_utils import FastaTokenizer # Required if meta.pkl contains FastaTokenizer specific structure

def export_to_onnx(model, dummy_input, file_path="model.onnx", input_names=None, output_names=None, dynamic_axes=None):
    """Exports a PyTorch model to ONNX format."""
    if input_names is None:
        input_names = ['input_ids']
    if output_names is None:
        output_names = ['logits']
    if dynamic_axes is None:
        dynamic_axes = {'input_ids': {0: 'batch_size', 1: 'sequence_length'},
                        'logits': {0: 'batch_size', 1: 'sequence_length'}}
    
    model.eval()
    print(f"Exporting model to ONNX: {file_path}")
    try:
        # Ensure dummy_input is on the same device as the model (usually CPU for export)
        # model.to('cpu') should have been called before this if model was on GPU
        # dummy_input = dummy_input.to(next(model.parameters()).device) # Not strictly needed if model is already on CPU
        
        torch.onnx.export(
            model,
            dummy_input,
            file_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
            opset_version=14,  # Or a version compatible with your target runtime
            verbose=False # Set to True for detailed ONNX export logging
        )
        print("ONNX export successful.")
    except Exception as e:
        print(f"ONNX export failed: {e}")
        # import traceback
        # traceback.print_exc() # For more detailed error
        raise # Re-raise exception to halt if critical

def export_for_transformers_js(model, config, tokenizer, save_directory):
    """
    Exports the model to a format compatible with Transformers.js (Hugging Face format).
    """
    print(f"Attempting to export model for Transformers.js to {save_directory}")
    os.makedirs(save_directory, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))

    hf_config_data = {
        "model_type": "gpt2", 
        "vocab_size": config.vocab_size,
        "n_positions": config.block_size,
        "n_embd": config.n_embd,
        "n_layer": config.n_layer,
        "n_head": config.n_head,
        "activation_function": "gelu_new", 
        "resid_pdrop": config.dropout, 
        "embd_pdrop": config.dropout,  
        "attn_pdrop": config.dropout,  
        "layer_norm_epsilon": 1e-5, 
        "initializer_range": 0.02, 
        "bos_token_id": tokenizer.bos_id if hasattr(tokenizer, 'bos_id') else None,
        "eos_token_id": tokenizer.eos_id if hasattr(tokenizer, 'eos_id') else None,
        "pad_token_id": tokenizer.pad_id if hasattr(tokenizer, 'pad_id') else None,
        "unk_token_id": tokenizer.unk_id if hasattr(tokenizer, 'unk_id') else None,
    }
    with open(os.path.join(save_directory, "config.json"), 'w') as f:
        json.dump(hf_config_data, f, indent=4)
    print("Saved Hugging Face style config.json.")

    if hasattr(tokenizer, 'stoi') and hasattr(tokenizer, 'itos'):
        vocab_json_path = os.path.join(save_directory, "vocab.json")
        # HF vocab.json is token -> id
        with open(vocab_json_path, 'w', encoding='utf-8') as f:
            json.dump(tokenizer.stoi, f, ensure_ascii=False, indent=2)
        print(f"Saved vocab.json to {vocab_json_path}")
        
        # Create merges.txt (empty for char-level, but sometimes expected)
        with open(os.path.join(save_directory, "merges.txt"), 'w') as f:
            pass # Empty for char tokenizers like FastaTokenizer
        print("Saved empty merges.txt (char-level tokenizer).")

        tokenizer_config_data = {
            "tokenizer_class": "GPT2Tokenizer", # Or PreTrainedTokenizerFast if more appropriate
            "model_max_length": config.block_size,
            "bos_token": tokenizer.decode([tokenizer.bos_id]) if hasattr(tokenizer, 'bos_id') and hasattr(tokenizer, 'decode') else "[BOS]",
            "eos_token": tokenizer.decode([tokenizer.eos_id]) if hasattr(tokenizer, 'eos_id') and hasattr(tokenizer, 'decode') else "[EOS]",
            "unk_token": tokenizer.decode([tokenizer.unk_id]) if hasattr(tokenizer, 'unk_id') and hasattr(tokenizer, 'decode') else "[UNK]",
            "pad_token": tokenizer.decode([tokenizer.pad_id]) if hasattr(tokenizer, 'pad_id') and hasattr(tokenizer, 'decode') else "[PAD]",
            # Additional special tokens if any:
            "additional_special_tokens": [
                tok_str for tok_str in [
                    tokenizer.decode([getattr(tokenizer, t, -1)]) if hasattr(tokenizer, t) and hasattr(tokenizer, 'decode') else None
                    for t in ['sep_id', 'dna_marker_id', 'rna_marker_id', 'protein_marker_id', 'desc_marker_id']
                ] if tok_str is not None and tok_str not in ["[BOS]", "[EOS]", "[UNK]", "[PAD]"]
            ]

        }
        # Filter out None values from special tokens before saving
        for key in ["bos_token", "eos_token", "unk_token", "pad_token"]:
            if tokenizer_config_data[key] is None:
                del tokenizer_config_data[key]
        tokenizer_config_data["additional_special_tokens"] = [t for t in tokenizer_config_data["additional_special_tokens"] if t]


        with open(os.path.join(save_directory, "tokenizer_config.json"), 'w') as f:
            json.dump(tokenizer_config_data, f, indent=4)
        print("Saved tokenizer_config.json.")
    else:
        print("Tokenizer does not have 'stoi'/'itos' attributes, cannot save vocab.json automatically for Transformers.js.")

    print("Transformers.js export attempt complete. Manual verification and adjustments are likely needed.")

def export_for_tensorflow_js(onnx_model_path, base_save_directory):
    """
    Converts an ONNX model to TensorFlow.js format using tfjs-converter.
    Requires `tensorflowjs_converter` to be installed.
    base_save_directory: The directory where the 'tfjs_model' subdirectory will be created containing the model.
    """
    print(f"Attempting to convert ONNX model {onnx_model_path} to TensorFlow.js format.")
    
    # The converter will place model.json and shards inside this specific directory.
    tfjs_model_target_dir = os.path.join(base_save_directory, "tfjs_model")
    os.makedirs(tfjs_model_target_dir, exist_ok=True) # Ensure this target directory exists

    print(f"Target TensorFlow.js model directory: {tfjs_model_target_dir}")

    try:
        command = [
            "tensorflowjs_converter",
            "--input_format=onnx",
            "--output_format=tfjs_graph_model",
            onnx_model_path, # Input ONNX model file
            tfjs_model_target_dir  # Output directory for TFJS model files
        ]
        print(f"Running command: {' '.join(command)}")
        
        # Capture output for better diagnostics
        result = subprocess.run(command, check=False, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"Error during TensorFlow.js conversion (return code {result.returncode}):")
            if result.stdout: print("STDOUT:\n", result.stdout)
            if result.stderr: print("STDERR:\n", result.stderr)
            # Optional: raise RuntimeError to make it a hard fail
            # raise RuntimeError(f"TensorFlow.js converter failed with return code {result.returncode}. STDERR: {result.stderr}")
            return # Exit function on failure

        # Check if usage message is in output, which can happen even with exit code 0 if args are slightly off
        if "usage: TensorFlow.js model converters" in result.stdout or \
           "usage: TensorFlow.js model converters" in result.stderr:
            print("TensorFlow.js converter printed usage message, an argument might be incorrect or an internal issue occurred.")
            if result.stdout: print("STDOUT:\n", result.stdout)
            # stderr might have already been printed if returncode was non-zero
            if result.returncode == 0 and result.stderr: print("STDERR:\n", result.stderr)
            # raise RuntimeError("TensorFlow.js converter failed with usage message.")
            return

        print(f"TensorFlow.js model conversion process completed. Check logs. Model expected in {tfjs_model_target_dir}")
        if result.stdout: print("STDOUT (may contain success messages or warnings):\n", result.stdout)
        if result.stderr and result.returncode == 0: # Print stderr if any, even on nominal success (for TF warnings)
            print("STDERR (may contain warnings):\n", result.stderr)

    except FileNotFoundError:
        print("Error: tensorflowjs_converter not found. Please install it: pip install tensorflowjs")
    except Exception as e: # Catch any other unexpected errors
        print(f"An unexpected error occurred during TensorFlow.js conversion: {e}")
        # import traceback
        # traceback.print_exc()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Export nanoGPT model to various formats.")
    parser.add_argument("--ckpt_path", type=str, default="out-fasta/ckpt.pt", help="Path to the model checkpoint (.pt file).")
    parser.add_argument("--out_dir", type=str, default="out-fasta-export", help="Directory to save exported models.")
    parser.add_argument("--export_onnx", action="store_true", help="Export to ONNX.")
    parser.add_argument("--export_transformers_js", action="store_true", help="Export for Transformers.js.")
    parser.add_argument("--export_tfjs", action="store_true", help="Export to TensorFlow.js (requires ONNX and tfjs-converter).")
    
    args = parser.parse_args()

    if not os.path.exists(args.ckpt_path):
        print(f"Checkpoint path {args.ckpt_path} does not exist.")
        exit(1)

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading checkpoint from {args.ckpt_path}")
    checkpoint = torch.load(args.ckpt_path, map_location='cpu') 
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.eval() # Set model to evaluation mode

    tokenizer = None
    meta_path_ckpt = os.path.join(os.path.dirname(args.ckpt_path), 'meta.pkl')
    if 'tokenizer_meta' in checkpoint and checkpoint['tokenizer_meta'].get('vocab'):
        print("Loading tokenizer_meta from checkpoint.")
        tokenizer = FastaTokenizer(vocab=checkpoint['tokenizer_meta']['vocab'])
        for key, value in checkpoint['tokenizer_meta'].items():
            if key not in ['vocab', 'stoi', 'itos']: # vocab, stoi, itos are handled by constructor or properties
                 setattr(tokenizer, key, value)
    elif os.path.exists(meta_path_ckpt):
        print(f"Loading tokenizer metadata from {meta_path_ckpt}")
        tokenizer = FastaTokenizer.from_file(meta_path_ckpt)
    else:
        print("Warning: Tokenizer metadata (meta.pkl or 'tokenizer_meta' in checkpoint) not found. Transformers.js export might be incomplete for tokenizer.")

    dummy_input_seq_len = min(128, gptconf.block_size) 
    dummy_input = torch.randint(0, gptconf.vocab_size, (1, dummy_input_seq_len), dtype=torch.long)

    onnx_model_path = os.path.join(args.out_dir, "model.onnx")

    if args.export_onnx or args.export_tfjs: # TFJS export depends on ONNX
        if not os.path.exists(onnx_model_path) or args.export_onnx : # Export if specifically requested or if needed for TFJS and not exists
            try:
                export_to_onnx(model, dummy_input, onnx_model_path)
            except Exception as e:
                print(f"Halting due to ONNX export failure: {e}")
                exit(1) # Stop if ONNX export fails and is needed
        elif not args.export_onnx and args.export_tfjs and os.path.exists(onnx_model_path):
            print(f"Using existing ONNX model for TFJS conversion: {onnx_model_path}")


    if args.export_transformers_js:
        if tokenizer:
            transformers_js_dir = os.path.join(args.out_dir, "transformers_js_model")
            export_for_transformers_js(model, gptconf, tokenizer, transformers_js_dir)
        else:
            print("Skipping Transformers.js export: tokenizer not available.")
            
    if args.export_tfjs:
        if os.path.exists(onnx_model_path):
            # The base directory for TFJS output (e.g., out-fasta-export/tfjs_artifacts)
            tfjs_base_dir = os.path.join(args.out_dir, "tfjs_artifacts") 
            os.makedirs(tfjs_base_dir, exist_ok=True)
            export_for_tensorflow_js(onnx_model_path, tfjs_base_dir)
        else:
            print("Skipping TensorFlow.js export: ONNX model could not be created/found.")
            
    print(f"Export process finished. Check {args.out_dir} for outputs.")