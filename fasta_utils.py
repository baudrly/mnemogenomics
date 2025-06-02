import re
import os
import pickle
import numpy as np
from collections import Counter

# Define character sets and special tokens
DNA_CHARS = "ACGTN"
RNA_CHARS = "ACGUN" # N for unknown
PROTEIN_CHARS_EXTENDED = "ARNDCQEGHILKMFPSTWYVBZXJO" # B,Z,X ambiguous, J either Leu/Ile, O Pyrrolysine, U Selenocysteine

PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"
BOS_TOKEN = "[BOS]" # Beginning of sequence/string
EOS_TOKEN = "[EOS]" # End of sequence/string
SEP_TOKEN = "[SEP]" # Separator between header and sequence
DNA_MARKER_TOKEN = "[DNA]"
RNA_MARKER_TOKEN = "[RNA]"
PROTEIN_MARKER_TOKEN = "[PROTEIN]"
DESC_MARKER_TOKEN = "[DESC]" # Marks the beginning of a description part

SPECIAL_TOKENS = [
    PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN, SEP_TOKEN,
    DNA_MARKER_TOKEN, RNA_MARKER_TOKEN, PROTEIN_MARKER_TOKEN, DESC_MARKER_TOKEN
]

# Heuristic: characters likely to be in descriptions (extend as needed)
DESCRIPTION_CHARS = "abcdefghijklmnopqrstuvwxyz0123456789 .,:;-_()[]{}?/\\!@#$%^&*+='\""

class FastaTokenizer:
    def __init__(self, vocab=None, vocab_file=None):
        if vocab:
            self.vocab = vocab
        elif vocab_file and os.path.exists(vocab_file):
            with open(vocab_file, 'rb') as f:
                saved_vocab_data = pickle.load(f)
                self.vocab = saved_vocab_data['vocab']
        else:
            self.vocab = []

        if not self.vocab: # Build default vocab if none provided/loaded
            self.vocab.extend(SPECIAL_TOKENS)
            self.vocab.extend(list(DNA_CHARS))
            self.vocab.extend(list(RNA_CHARS)) # Will have duplicate A,C,G,N
            self.vocab.extend(list(PROTEIN_CHARS_EXTENDED))
            self.vocab.extend(list(DESCRIPTION_CHARS))
            self.vocab = sorted(list(set(self.vocab))) # Remove duplicates and sort

        self.stoi = {ch: i for i, ch in enumerate(self.vocab)}
        self.itos = {i: ch for i, ch in enumerate(self.vocab)}
        
        # Ensure special tokens are correctly mapped
        for token in SPECIAL_TOKENS:
            if token not in self.stoi:
                # This case should not happen if vocab is built correctly
                new_idx = len(self.vocab)
                self.vocab.append(token)
                self.stoi[token] = new_idx
                self.itos[new_idx] = token
        
        self.pad_id = self.stoi[PAD_TOKEN]
        self.unk_id = self.stoi[UNK_TOKEN]
        self.bos_id = self.stoi[BOS_TOKEN]
        self.eos_id = self.stoi[EOS_TOKEN]
        self.sep_id = self.stoi[SEP_TOKEN]
        self.dna_marker_id = self.stoi[DNA_MARKER_TOKEN]
        self.rna_marker_id = self.stoi[RNA_MARKER_TOKEN]
        self.protein_marker_id = self.stoi[PROTEIN_MARKER_TOKEN]
        self.desc_marker_id = self.stoi[DESC_MARKER_TOKEN]


    @property
    def vocab_size(self):
        return len(self.vocab)

    def encode(self, s, is_sequence_data=False):
        if is_sequence_data:
            s = s.upper() # Sequences are typically uppercase
        else:
            s = s.lower() # Descriptions to lowercase
        
        tokens = []
        for char in s:
            if char in self.stoi:
                tokens.append(self.stoi[char])
            elif is_sequence_data and char in DNA_CHARS+RNA_CHARS+PROTEIN_CHARS_EXTENDED: # Should be caught by vocab
                tokens.append(self.stoi.get(char.upper(), self.unk_id))
            else: # Character not in vocab (e.g. rare punctuation in desc)
                 tokens.append(self.unk_id)
        return tokens

    def decode(self, ids):
        return "".join([self.itos.get(i, UNK_TOKEN) for i in ids])

    def save_vocab(self, filepath):
        # Save vocab and special token mappings
        vocab_data = {
            'vocab': self.vocab,
            'stoi': self.stoi,
            'itos': self.itos,
            'pad_id': self.pad_id,
            'unk_id': self.unk_id,
            'bos_id': self.bos_id,
            'eos_id': self.eos_id,
            'sep_id': self.sep_id,
            'dna_marker_id': self.dna_marker_id,
            'rna_marker_id': self.rna_marker_id,
            'protein_marker_id': self.protein_marker_id,
            'desc_marker_id': self.desc_marker_id,
        }
        with open(filepath, 'wb') as f:
            pickle.dump(vocab_data, f)

    @classmethod
    def from_file(cls, filepath):
        with open(filepath, 'rb') as f:
            vocab_data = pickle.load(f)
        tokenizer = cls(vocab=vocab_data['vocab']) # Re-init to set up stoi/itos correctly
        # Ensure all loaded attributes are set
        for key, value in vocab_data.items():
            if key not in ['vocab', 'stoi', 'itos']: # vocab, stoi, itos are handled by constructor
                 setattr(tokenizer, key, value)
        return tokenizer


def clean_header(header):
    """
    Cleans a FASTA header to extract a functional description.
    Removes accession numbers and common non-descriptive terms.
    """
    if header.startswith(">"):
        header = header[1:]

    # Attempt to remove common accession patterns at the beginning
    # This is a heuristic and might need adjustment
    patterns = [
        r"^[A-Z0-9]{1,4}\|[A-Z0-9_\.-]+\|[A-Z0-9_\.-]+\s+",  # SwissProt/TrEMBL like sp|P12345|GENE_HUMAN
        r"^[a-zA-Z]{2,3}_?[0-9]{5,}(\.[0-9]+)?\s+",      # GenBank/RefSeq like NM_000014.6 or XP_0012345.1
        r"^[A-Z0-9]{6,10}(\.[0-9]+)?\s+",                # Generic accession like P12345 or ABC123456
        r"^lcl\|[^\s]+\s+",                              # Local NCBI accession
        r"^gi\|[0-9]+\|[a-z]+\|[A-Z0-9\._]+\|[^\s]*\s*", # Old GI format
    ]
    original_header = header
    for pattern in patterns:
        match = re.match(pattern, header)
        if match:
            header = header[match.end():] # Remove the matched part
            break # Assume first match is the primary ID

    # If no pattern matched, try removing the first "word" if it looks like an ID
    if header == original_header:
        parts = header.split(" ", 1)
        if len(parts) > 1 and (re.match(r"^[A-Za-z0-9_\.-]+$", parts[0]) and any(c.isdigit() for c in parts[0])):
            header = parts[1]


    # Remove common non-functional phrases (case-insensitive)
    non_functional_phrases = [
        r"transcript variant \d+", r"isoform \d+", r"partial cds", r"complete cds",
        r"predicted protein", r"hypothetical protein", r"unknown function",
        r"low-quality protein", r"PREDICTED:", r"similar to", r"putative",
        r"mrna sequence", r"cdna sequence", r"dna sequence", r"genomic sequence",
        r"whole genome shotgun sequence", r"complete genome", r"chromosome \w+", r"contig \w+",
        r"unnamed protein product", r"uncharacterized protein",
    ]
    for phrase in non_functional_phrases:
        header = re.sub(r"\b" + phrase + r"\b", "", header, flags=re.IGNORECASE)

    # Standardize: lowercase, strip whitespace, reduce multiple spaces
    header = header.lower()
    header = re.sub(r"\s+", " ", header).strip()
    
    # If header becomes too short or empty, revert to a simpler cleaning
    if not header or len(header) < 5 : # arbitrary length threshold
        h_parts = original_header.split(" ", 1)
        if len(h_parts) > 1:
            header = h_parts[1].lower().strip()
            header = re.sub(r"\s+", " ", header).strip()
        else: # if only one word, use it
            header = original_header.lower().strip()

    return header if header else "unknown function"


def detect_sequence_type(sequence):
    """
    Detects if a sequence is DNA, RNA, or Protein.
    """
    seq_upper = sequence.upper()
    counts = Counter(seq_upper)
    length = len(seq_upper)
    if length == 0:
        return "UNKNOWN"

    # Character sets
    dna_bases = set(DNA_CHARS)
    rna_bases = set(RNA_CHARS)
    protein_aas = set(PROTEIN_CHARS_EXTENDED)

    # Check for RNA: presence of 'U' and no 'T' (unless 'T' is a tiny fraction)
    if counts['U'] > 0:
        if counts['T'] == 0 or (counts['T'] / length < 0.01 and counts['U'] > counts['T']): # Allow tiny T if U is dominant
            # Check if it's not predominantly protein characters
            protein_char_count = sum(counts[aa] for aa in protein_aas if aa not in rna_bases)
            if protein_char_count / length < 0.1: # If less than 10% are protein-specific chars
                return "RNA"

    # Check for DNA: presence of 'T' and no 'U' (unless 'U' is tiny fraction - e.g. sequencing error or U in DNA)
    if counts['T'] > 0:
        if counts['U'] == 0 or (counts['U'] / length < 0.01 and counts['T'] > counts['U']):
            protein_char_count = sum(counts[aa] for aa in protein_aas if aa not in dna_bases)
            if protein_char_count / length < 0.1:
                return "DNA"

    # If no U or T, or ambiguous, check for protein
    # Protein if high proportion of protein-specific characters
    # and low proportion of GACT (common in all)
    
    # Count characters unique to protein vs DNA/RNA typical alphabet
    protein_specific_chars = protein_aas - dna_bases - rna_bases # Chars like L,K,M,F,P,S,W,Y,V etc.
    
    # Count characters typically DNA/RNA
    nucleic_chars = dna_bases.union(rna_bases) - protein_aas # Chars like T, U (if not ambiguous protein codes)

    protein_evidence = sum(counts[aa] for aa in protein_specific_chars)
    nucleic_evidence = sum(counts[base] for base in nucleic_chars) # Sum of T, U primarily
    
    # Count ambiguous chars that are in all sets (A,C,G,N etc.)
    # For ambiguous like 'N', 'X', they don't help distinguish much here
    
    # If sequence only contains DNA characters (ACGTN)
    if all(c in dna_bases for c in counts):
        return "DNA"
    # If sequence only contains RNA characters (ACGUN)
    if all(c in rna_bases for c in counts):
        return "RNA"
        
    # If more protein-specific characters than T/U, likely protein
    if protein_evidence > nucleic_evidence and protein_evidence / length > 0.5: # More than 50% protein-specific chars
        return "PROTEIN"

    # Fallback: if mostly protein alphabet
    protein_chars_count = sum(counts[aa] for aa in protein_aas)
    if protein_chars_count / length > 0.85: # If >85% of chars are in protein alphabet
        return "PROTEIN"

    # Default to DNA if ambiguous but uses DNA alphabet primarily
    if all(c in dna_bases or c in protein_aas for c in counts) and counts.get('T',0) > 0:
         return "DNA" # Common if a protein coding gene (DNA) is provided

    return "UNKNOWN" # If truly ambiguous or mixed

def parse_fasta(file_path, num_entries=None):
    """
    Parses a FASTA file.
    Yields (header, sequence) tuples.
    """
    records = []
    current_sequence_lines = []
    current_header = None
    count = 0
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_header is not None:
                    records.append((current_header, "".join(current_sequence_lines)))
                    current_sequence_lines = []
                    count += 1
                    if num_entries is not None and count >= num_entries:
                        return records # Return list of records
                current_header = line
            else:
                if current_header is not None: # Ensure we are inside a record
                    current_sequence_lines.append(line)
        
        # Add the last record
        if current_header is not None:
            records.append((current_header, "".join(current_sequence_lines)))
    return records # Return list of records