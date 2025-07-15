"""
Adaptive data preparation for Shakespeare dataset.
Automatically selects optimal vocabulary based on hardware capabilities.
"""

import os
import pickle
import requests
import numpy as np
import torch
import tiktoken

def download_shakespeare():
    """Download Shakespeare dataset if not present."""
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
    
    if not os.path.exists(input_file_path):
        print(f"Downloading Shakespeare dataset to {input_file_path}")
        with open(input_file_path, 'w', encoding='utf-8') as f:
            f.write(requests.get(data_url).text)
    
    return input_file_path

def prepare_data():
    """Prepare Shakespeare data with adaptive vocabulary selection."""
    
    # Check hardware capabilities
    use_gpu = torch.cuda.is_available()
    print(f"GPU available: {use_gpu}")
    if use_gpu:
        gpu_props = torch.cuda.get_device_properties(0)
        print(f"GPU: {gpu_props.name} (Compute Capability: {gpu_props.major}.{gpu_props.minor})")
    
    # Download Shakespeare text
    input_file_path = download_shakespeare()
    
    with open(input_file_path, 'r', encoding='utf-8') as f:
        data = f.read()
    
    print(f"Shakespeare text length: {len(data):,} characters")
    
    # Initialize tokenizer
    enc = tiktoken.get_encoding("gpt2")
    
    # Tokenize the data
    tokens = enc.encode(data)
    print(f"Total tokens: {len(tokens):,}")
    
    if use_gpu:
        # GPU configuration: Use full vocabulary for optimal performance
        vocab_size = enc.n_vocab  # 50257
        print(f"Using full vocabulary for GPU: {vocab_size} tokens")
        
        # Use all tokens as-is
        final_tokens = tokens
        
        # Create token mappings (identity mapping)
        token_to_compressed = {i: i for i in range(vocab_size)}
        compressed_to_token = {i: i for i in range(vocab_size)}
        
    else:
        # CPU configuration: Use compressed vocabulary for memory efficiency
        print("Creating compressed vocabulary for CPU...")
        
        # Count token frequencies
        token_counts = {}
        for token in tokens:
            token_counts[token] = token_counts.get(token, 0) + 1
        
        # Sort by frequency and take top tokens
        sorted_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Take top 11,706 tokens (optimized for CPU memory constraints)
        compressed_vocab_size = 11706
        top_tokens = [token for token, count in sorted_tokens[:compressed_vocab_size]]
        
        # Create mapping from original to compressed tokens
        token_to_compressed = {token: i for i, token in enumerate(top_tokens)}
        compressed_to_token = {i: token for i, token in enumerate(top_tokens)}
        
        # Add unknown token for rare tokens
        unk_token_id = compressed_vocab_size - 1
        
        # Convert tokens to compressed vocabulary
        final_tokens = []
        for token in tokens:
            if token in token_to_compressed:
                final_tokens.append(token_to_compressed[token])
            else:
                final_tokens.append(unk_token_id)
        
        vocab_size = compressed_vocab_size
        print(f"Using compressed vocabulary for CPU: {vocab_size} tokens")
        print(f"Unknown token coverage: {(len(tokens) - len([t for t in tokens if t in token_to_compressed])) / len(tokens) * 100:.2f}%")
    
    # Convert to numpy array
    final_tokens = np.array(final_tokens, dtype=np.uint16)
    
    # Split into train/val
    n = len(final_tokens)
    train_tokens = final_tokens[:int(n*0.9)]
    val_tokens = final_tokens[int(n*0.9):]
    
    # Save binary files
    train_path = os.path.join(os.path.dirname(__file__), 'train.bin')
    val_path = os.path.join(os.path.dirname(__file__), 'val.bin')
    
    train_tokens.tofile(train_path)
    val_tokens.tofile(val_path)
    
    print(f"Train tokens: {len(train_tokens):,}")
    print(f"Val tokens: {len(val_tokens):,}")
    
    # Save metadata
    meta = {
        'vocab_size': vocab_size,
        'use_gpu': use_gpu,
        'original_vocab_size': enc.n_vocab,
        'compression_ratio': vocab_size / enc.n_vocab if not use_gpu else 1.0,
        'token_to_compressed': token_to_compressed if not use_gpu else None,
        'compressed_to_token': compressed_to_token if not use_gpu else None,
    }
    
    meta_path = os.path.join(os.path.dirname(__file__), 'meta.pkl')
    with open(meta_path, 'wb') as f:
        pickle.dump(meta, f)
    
    print(f"Saved metadata to {meta_path}")
    print(f"Final vocabulary size: {vocab_size}")
    print("Data preparation complete!")

if __name__ == '__main__':
    prepare_data()
