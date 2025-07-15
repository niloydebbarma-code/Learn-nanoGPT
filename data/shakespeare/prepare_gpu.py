import os
import requests
import tiktoken
import numpy as np
import pickle
import torch

# download the tiny shakespeare dataset
input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
if not os.path.exists(input_file_path):
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    with open(input_file_path, 'w', encoding='utf-8') as f:
        f.write(requests.get(data_url).text)

with open(input_file_path, 'r', encoding='utf-8') as f:
    data = f.read()
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# Initialize GPT-2 tokenizer
enc = tiktoken.get_encoding("gpt2")

# Check if GPU is available to decide on vocabulary compression
use_full_vocab = torch.cuda.is_available()

if use_full_vocab:
    print("GPU detected - using full GPT-2 vocabulary")
    
    # Use full vocabulary for GPU training
    train_ids = enc.encode_ordinary(train_data)
    val_ids = enc.encode_ordinary(val_data)
    
    # Save metadata for full vocabulary
    meta = {
        'vocab_size': enc.n_vocab,
        'itos': {i: enc.decode([i]) for i in range(enc.n_vocab)},
        'stoi': {enc.decode([i]): i for i in range(enc.n_vocab)},
    }
    
    print(f"Full vocabulary size: {enc.n_vocab}")
    print(f"train has {len(train_ids):,} tokens")
    print(f"val has {len(val_ids):,} tokens")
    
else:
    print("CPU detected - using compressed vocabulary")
    
    # Tokenize the Shakespeare text to find used tokens
    train_tokens = enc.encode_ordinary(train_data)
    val_tokens = enc.encode_ordinary(val_data)
    all_tokens = train_tokens + val_tokens

    # Create compressed vocabulary using only tokens found in Shakespeare text
    unique_tokens = sorted(set(all_tokens))
    print(f"Original vocabulary size: {enc.n_vocab}")
    print(f"Unique tokens in Shakespeare: {len(unique_tokens)}")

    # Create mapping from original token IDs to compressed token IDs
    original_to_compressed = {orig: comp for comp, orig in enumerate(unique_tokens)}
    compressed_to_original = {comp: orig for orig, comp in original_to_compressed.items()}

    # Apply compression to token sequences
    train_ids = [original_to_compressed[token] for token in train_tokens]
    val_ids = [original_to_compressed[token] for token in val_tokens]

    print(f"Compressed vocabulary size: {len(unique_tokens)}")
    print(f"Memory reduction: {(1 - len(unique_tokens) / enc.n_vocab) * 100:.1f}%")
    
    # Save metadata for compressed vocabulary
    meta = {
        'vocab_size': len(unique_tokens),
        'itos': {i: enc.decode([compressed_to_original[i]]) for i in range(len(unique_tokens))},
        'stoi': {enc.decode([compressed_to_original[i]]): i for i in range(len(unique_tokens))},
        'original_to_compressed': original_to_compressed,
        'compressed_to_original': compressed_to_original,
    }

# Save metadata
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

# Convert to numpy arrays and save as binary files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

print(f"train.bin has {len(train_ids):,} tokens")
print(f"val.bin has {len(val_ids):,} tokens")
print(f"Saved {'full' if use_full_vocab else 'compressed'} vocabulary and binary data files")
