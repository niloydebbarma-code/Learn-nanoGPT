"""
Training script for Shakespeare text generation using nanoGPT.
Automatically detects and optimizes for available hardware (CPU/GPU).
To run: python train.py
"""

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch

from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# default config values optimized for Shakespeare training
# I/O
out_dir = 'out-shakespeare'
eval_interval = 250
log_interval = 10
eval_iters = 50 if torch.cuda.is_available() else 5  # Evaluation iterations
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = False # Save only when validation improves
init_from = 'scratch' # 'scratch' or 'resume'
# wandb logging - disabled for simplicity
wandb_log = False
wandb_project = 'shakespeare'
wandb_run_name = 'mini-gpt'
# data
dataset = 'shakespeare'
gradient_accumulation_steps = 8 if torch.cuda.is_available() else 4  # Gradient accumulation steps
batch_size = 16 if torch.cuda.is_available() else 16  # Batch size for training
block_size = 256 if torch.cuda.is_available() else 256  # Context length for training
# baby GPT model optimized for Shakespeare (can be upgraded for GPU)
n_layer = 6 if torch.cuda.is_available() else 6   # Number of transformer layers
n_head = 6 if torch.cuda.is_available() else 6    # Number of attention heads
n_embd = 384 if torch.cuda.is_available() else 384 # Embedding dimensions
dropout = 0.2
bias = True # use bias for better performance on small datasets
# adamw optimizer
learning_rate = 3e-4 # lower LR for stable training
max_iters = 1000  # Reduced for small dataset
weight_decay = 1e-2
beta1 = 0.9
beta2 = 0.99 # bigger beta2 for small token batches
grad_clip = 1.0
# learning rate decay settings
decay_lr = True
warmup_iters = 200
lr_decay_iters = 1000 # equal to max_iters
min_lr = 3e-5 # learning_rate / 10
# system
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Use float16 for T4 GPU (sm_75) compatibility, bfloat16 for newer GPUs (sm_80+)
if torch.cuda.is_available():
    # Check GPU compute capability
    if torch.cuda.get_device_properties(0).major >= 8:
        dtype = 'bfloat16'  # A100, H100, etc.
    else:
        dtype = 'float16'   # T4, V100, etc.
else:
    dtype = 'float32'       # CPU
compile = False  # Disable compilation for better compatibility
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
master_process = True
tokens_per_iter = gradient_accumulation_steps * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337)

# Hardware optimization
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    # Enable memory efficient attention
    torch.backends.cuda.enable_flash_sdp(True)

device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# poor man's data loader for Shakespeare datasets
data_dir = os.path.join('data', dataset)
def get_batch(split):
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init for Shakespeare character-level training
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout)
if init_from == 'scratch':
    print("Initializing a new model from scratch")
    # Use vocab size from Shakespeare dataset
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 65
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
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
# crop down the model block size if desired
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size
model.to(device)

# Compile the model for GPU optimization (disabled for compatibility)
if compile and device_type == 'cuda':
    print("Compiling model for GPU optimization...")
    try:
        model = torch.compile(model)
    except Exception as e:
        print(f"Model compilation failed: {e}")
        print("Continuing without compilation...")

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    print("Starting loss estimation...")
    out = {}
    model.eval()
    for split in ['train', 'val']:
        print(f"Evaluating {split} split...")
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            if k % 5 == 0:
                print(f"  Batch {k}/{eval_iters}")
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
        print(f"  {split} loss: {out[split]:.4f}")
    model.train()
    print("Loss estimation complete")
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging - disabled for simplicity
# No wandb logging for Shakespeare training

# training loop
print("Starting training loop...")
X, Y = get_batch('train') # fetch the very first batch
print("First batch fetched")
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model # no DDP wrapper for CPU training
running_mfu = -1.0
print("Entering main training loop...")
while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        print(f"Starting evaluation at iteration {iter_num}...")
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    if iter_num == 0 and eval_only:
        break

    # forward backward update for Shakespeare training
    for micro_step in range(gradient_accumulation_steps):
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps
        X, Y = get_batch('train')
        loss.backward()
        
        # Clear cache between micro steps for memory efficiency
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # clip the gradient
    if grad_clip != 0.0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    
    # step the optimizer
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5:
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
    
    iter_num += 1
    local_iter_num += 1
    
    if iter_num % 50 == 0:
        print(f"Completed {iter_num} iterations so far...")

    # termination conditions
    if iter_num > max_iters:
        print(f"Training completed! Reached max_iters={max_iters}")
        break


