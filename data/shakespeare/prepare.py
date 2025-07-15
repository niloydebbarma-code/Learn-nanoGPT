import os
import requests
import tiktoken
import numpy as np
import pickle

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

# encode with tiktoken gpt2 bpe
enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# create a smaller vocabulary with only tokens used in Shakespeare dataset
all_ids = train_ids + val_ids
unique_tokens = sorted(set(all_ids))
vocab_size = len(unique_tokens)
print(f"unique tokens in dataset: {vocab_size}")

# create mapping from original token ids to compressed ids
token_to_id = {token: i for i, token in enumerate(unique_tokens)}
id_to_token = {i: token for i, token in enumerate(unique_tokens)}

# remap the token ids to use smaller vocabulary
train_ids_remapped = [token_to_id[token] for token in train_ids]
val_ids_remapped = [token_to_id[token] for token in val_ids]

# export to bin files with remapped ids
train_ids = np.array(train_ids_remapped, dtype=np.uint16)
val_ids = np.array(val_ids_remapped, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# save the meta information as well, to help us encode/decode later
meta = {
    'vocab_size': vocab_size,
    'token_to_id': token_to_id,
    'id_to_token': id_to_token,
    'original_vocab_size': enc.n_vocab,
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

# train.bin has 301,966 tokens
# val.bin has 36,059 tokens
