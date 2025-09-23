import os
import pickle
import numpy as np

# Path to enwik8 (should be in same folder as this script)
input_file_path = os.path.join(os.path.dirname(__file__), 'enwik8')

with open(input_file_path, 'rb') as f:
    data = f.read().decode('utf-8', errors='ignore')

print(f"Total dataset length: {len(data):,} characters")

chars = sorted(list(set(data)))
vocab_size = len(chars)
print("Unique characters:")
print(''.join(chars))
print(f"Vocab size: {vocab_size:,}")

stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
def encode(s): return [stoi[c] for c in s]
def decode(l): return ''.join([itos[i] for i in l])

# 90% train, 5% val, 5% test
n = len(data)
train_end = int(n * 0.90)
val_end = int(n * 0.95)

train_data = data[:train_end]
val_data = data[train_end:val_end]
test_data = data[val_end:]

train_ids = np.array(encode(train_data), dtype=np.uint16)
val_ids = np.array(encode(val_data), dtype=np.uint16)
test_ids = np.array(encode(test_data), dtype=np.uint16)

print(f"Train tokens: {len(train_ids):,}")
print(f"Val tokens:   {len(val_ids):,}")
print(f"Test tokens:  {len(test_ids):,}")

meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}

out_dir = os.path.dirname(__file__)
train_ids.tofile(os.path.join(out_dir, 'train.bin'))
val_ids.tofile(os.path.join(out_dir, 'val.bin'))
test_ids.tofile(os.path.join(out_dir, 'test.bin'))

with open(os.path.join(out_dir, 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)
