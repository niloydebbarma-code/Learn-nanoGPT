"""
Sample from a trained Shakespeare model
"""
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
init_from = 'resume' # Load from trained Shakespeare model checkpoint
out_dir = 'out-shakespeare' # Shakespeare model checkpoint directory
start = "\n" # Starting text for Shakespeare generation
num_samples = 10 # Number of Shakespeare text samples to generate
max_new_tokens = 500 # Length of each generated Shakespeare text
temperature = 0.8 # Controls randomness in Shakespeare text generation
top_k = 200 # Limits vocabulary for more coherent Shakespeare text
seed = 1337
device = 'cuda' if torch.cuda.is_available() else 'cpu' # Auto-detect GPU/CPU
# Precision settings for different hardware
if torch.cuda.is_available():
    if torch.cuda.get_device_properties(0).major >= 8:
        dtype = 'bfloat16'  # Modern GPUs
    else:
        dtype = 'float16'   # Older GPUs like T4
else:
    dtype = 'float32'       # CPU fallback
compile = False # Disabled for compatibility

# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
device_type = 'cuda' if 'cuda' in device else 'cpu' # For autocast context
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# Load Shakespeare model
if init_from == 'resume':
    # Load Shakespeare model from checkpoint
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)

model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # PyTorch 2.0 compilation

# Check for Shakespeare dataset encoding
load_meta = False
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']:
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)
    print(f"Looking for meta.pkl at: {meta_path}")
    print(f"Meta file exists: {load_meta}")
    
if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    print(f"Meta keys: {list(meta.keys())}")
    
    # Create encoder/decoder for different prepare file formats
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    
    # Check for adaptive prepare format (token_to_compressed/compressed_to_token)
    if 'token_to_compressed' in meta and 'compressed_to_token' in meta:
        token_to_compressed = meta['token_to_compressed']
        compressed_to_token = meta['compressed_to_token']
        use_gpu = meta.get('use_gpu', False)
        
        print(f"Using adaptive prepare format (GPU: {use_gpu})")
        print(f"Vocabulary size: {meta['vocab_size']}")
        
        if use_gpu:
            # GPU mode: identity mapping (no compression)
            def encode(s):
                return enc.encode_ordinary(s)
            
            def decode(tokens):
                return enc.decode(tokens)
        else:
            # CPU mode: compressed vocabulary
            def encode(s):
                gpt2_tokens = enc.encode_ordinary(s)
                return [token_to_compressed.get(token, len(token_to_compressed)-1) for token in gpt2_tokens]
            
            def decode(tokens):
                gpt2_tokens = [compressed_to_token.get(token, compressed_to_token.get(len(compressed_to_token)-1, 0)) for token in tokens]
                return enc.decode(gpt2_tokens)
    
    # Check for standard prepare format (token_to_id/id_to_token)
    elif 'token_to_id' in meta and 'id_to_token' in meta:
        token_to_id = meta['token_to_id']
        id_to_token = meta['id_to_token']
        print("Using standard prepare format")
        print(f"Vocabulary size: {len(token_to_id)}")
        
        def encode(s):
            gpt2_tokens = enc.encode_ordinary(s)
            return [token_to_id[token] for token in gpt2_tokens if token in token_to_id]
        
        def decode(tokens):
            gpt2_tokens = [id_to_token[token] for token in tokens if token in id_to_token]
            return enc.decode(gpt2_tokens)
    
    # Check for character-level format (stoi/itos)
    elif 'stoi' in meta and 'itos' in meta:
        stoi = meta['stoi']
        itos = meta['itos']
        print("Using character-level format")
        print(f"Vocabulary size: {len(stoi)}")
        
        def encode(s):
            return [stoi[c] for c in s if c in stoi]
        
        def decode(tokens):
            return ''.join([itos.get(token, '') for token in tokens])
    
    else:
        print("Meta structure not compatible - available keys:", list(meta.keys()))
        print("Expected keys: 'token_to_compressed'/'compressed_to_token' OR 'token_to_id'/'id_to_token' OR 'stoi'/'itos'")
        print("Cannot proceed without proper token encoding")
        exit(1)
else:
    print("No meta.pkl found or checkpoint config missing")
    print("Cannot proceed without Shakespeare encoding")
    exit(1)

# Encode starting text
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()
start_ids = encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

# Generate Shakespeare text
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            print(decode(y[0].tolist()))
            print('---------------')
