#!/usr/bin/env python3
import os
import random
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from fasta_utils import (
    FastaTokenizer, parse_fasta, clean_header, detect_sequence_type,
    BOS_TOKEN, EOS_TOKEN, SEP_TOKEN, DESC_MARKER_TOKEN,
    DNA_MARKER_TOKEN, RNA_MARKER_TOKEN, PROTEIN_MARKER_TOKEN
)

def prepare_data(fasta_file, out_dir, val_split=0.1, max_seq_len=1024, min_seq_len=10):
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading and parsing FASTA file: {fasta_file}")
    records = parse_fasta(fasta_file)
    print(f"Found {len(records)} records.")

    if not records:
        print("No records found. Exiting.")
        return

    # Initialize tokenizer (it will build a default vocab)
    tokenizer = FastaTokenizer()
    
    processed_data = []
    skipped_count = 0

    type_map = {
        "DNA": tokenizer.dna_marker_id,
        "RNA": tokenizer.rna_marker_id,
        "PROTEIN": tokenizer.protein_marker_id,
    }

    for header_raw, seq_raw in records:
        if not seq_raw or len(seq_raw) < min_seq_len :
            skipped_count += 1
            continue
            
        description = clean_header(header_raw)
        seq_type = detect_sequence_type(seq_raw)

        if seq_type == "UNKNOWN" or seq_type not in type_map:
            # Try to infer from header if possible, or skip
            if "dna" in description or "gene" in description: seq_type = "DNA"
            elif "rna" in description or "mrna" in description or "transcript" in description : seq_type = "RNA"
            elif "protein" in description or "peptide" in description: seq_type = "PROTEIN"
            else:
                skipped_count += 1
                continue # Skip if type cannot be determined

        type_marker_id = type_map[seq_type]
        desc_marker_id = tokenizer.desc_marker_id

        # Tokenize description and sequence
        # For sequences, truncate from the end if too long. For descriptions, truncate from end.
        tokenized_desc = tokenizer.encode(description, is_sequence_data=False)[:max_seq_len//2 - 5] # Reserve space for special tokens
        tokenized_seq = tokenizer.encode(seq_raw, is_sequence_data=True)[:max_seq_len//2 - 5]


        # Format 1: [BOS] [DESC_MARKER] <desc> [SEP] [TYPE_MARKER] <seq> [EOS]
        # Max length for one part is roughly max_seq_len / 2 - safety_margin_for_tokens
        
        # Recalculate available space based on actual tokenized lengths
        # Total special tokens: BOS, DESC_MARKER, SEP, TYPE_MARKER, EOS = 5 tokens
        # Available length for content = max_seq_len - 5
        
        available_content_len = max_seq_len - 5
        
        # If total length of desc + seq exceeds available, truncate intelligently
        if len(tokenized_desc) + len(tokenized_seq) > available_content_len:
            # Prioritize sequence, then description
            if len(tokenized_seq) > available_content_len * 0.6: # Sequence takes up to 60%
                tokenized_seq = tokenized_seq[:int(available_content_len * 0.6)]
            
            remaining_len_for_desc = available_content_len - len(tokenized_seq)
            if len(tokenized_desc) > remaining_len_for_desc:
                tokenized_desc = tokenized_desc[:remaining_len_for_desc]
        
        # Check again, if still too long, hard truncate sequence more
        if len(tokenized_desc) + len(tokenized_seq) > available_content_len:
             tokenized_seq = tokenized_seq[:available_content_len - len(tokenized_desc)]


        if not tokenized_desc or not tokenized_seq: # Skip if either part became empty
            skipped_count +=1
            continue

        # Construct samples
        # Sample 1: Description -> Sequence
        sample1 = [tokenizer.bos_id, desc_marker_id] + tokenized_desc + \
                  [tokenizer.sep_id, type_marker_id] + tokenized_seq + [tokenizer.eos_id]
        
        # Sample 2: Sequence -> Description
        sample2 = [tokenizer.bos_id, type_marker_id] + tokenized_seq + \
                  [tokenizer.sep_id, desc_marker_id] + tokenized_desc + [tokenizer.eos_id]
        
        if len(sample1) <= max_seq_len:
            processed_data.append(np.array(sample1, dtype=np.uint16))
        else:
            skipped_count +=1
            # print(f"Warning: Sample 1 too long ({len(sample1)} > {max_seq_len}). Skipping.")

        if len(sample2) <= max_seq_len:
            processed_data.append(np.array(sample2, dtype=np.uint16))
        else:
            skipped_count +=1
            # print(f"Warning: Sample 2 too long ({len(sample2)} > {max_seq_len}). Skipping.")

    print(f"Processed {len(processed_data)} samples. Skipped {skipped_count} records.")
    if not processed_data:
        print("No data to save after processing. Check input file and parameters.")
        return

    # Shuffle data
    random.shuffle(processed_data)

    # Split into train and validation
    train_data, val_data = train_test_split(processed_data, test_size=val_split, random_state=42)

    # Concatenate allids into one large array and save to .bin files
    train_ids = np.concatenate(train_data)
    val_ids = np.concatenate(val_data)

    print(f"Training set: {len(train_ids)} tokens")
    print(f"Validation set: {len(val_ids)} tokens")

    train_ids.tofile(os.path.join(out_dir, 'train.bin'))
    val_ids.tofile(os.path.join(out_dir, 'val.bin'))

    # Save metadata (tokenizer vocabulary and mappings)
    meta_path = os.path.join(out_dir, 'meta.pkl')
    tokenizer.save_vocab(meta_path)
    print(f"Vocabulary and metadata saved to {meta_path}")
    print(f"Vocabulary size: {tokenizer.vocab_size}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Prepare FASTA data for nanoGPT training.")
    parser.add_argument("fasta_file", type=str, help="Path to the input FASTA file.")
    parser.add_argument("--out_dir", type=str, default="data/fasta_prepared", help="Output directory for processed data.")
    parser.add_argument("--val_split", type=float, default=0.05, help="Fraction of data to use for validation (e.g., 0.05 for 5%).") # Smaller default for large datasets
    parser.add_argument("--max_seq_len", type=int, default=512, help="Maximum sequence length for each combined sample.") # Reduced default for faster processing
    parser.add_argument("--min_seq_len", type=int, default=20, help="Minimum raw sequence length to consider a record.")


    args = parser.parse_args()
    prepare_data(args.fasta_file, args.out_dir, args.val_split, args.max_seq_len, args.min_seq_len)
    print(f"Data preparation complete. Processed data saved in {args.out_dir}")
    print(f"You can now train a model using this data, e.g.:")
    print(f"python train_fasta.py --dataset=fasta_prepared ... (other training args)")