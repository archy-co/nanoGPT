import os
import pickle
import numpy as np
import torch
from contextlib import nullcontext
from model import GPTConfig, GPT

out_dirs = ["enwik8_2_babyPro", "enwik8_3_babyPro_ext0_forgetting0", "enwik8_4_babyPro_ext1_bLR0", "enwik8_5_babyPro_ext2_AttStr0", "enwik8_6_babyPro_ext1x2_forgetting_and_bLR0"]
dataset = 'enwik8'
split = 'test'
eval_iters = 200
batch_size = 32
block_size = 512
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device == 'cpu' else torch.amp.autocast(device_type='cuda', dtype=ptdtype)

N = 10   # number of trials

data_dir = os.path.join('data', dataset)
data_path = os.path.join(data_dir, f'{split}.bin')
data = np.memmap(data_path, dtype=np.uint16, mode='r')

def get_batch():
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

for out_dir in out_dirs:
    print(f"--------- {out_dir} ---------")
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    model_args = checkpoint['model_args']

    model = GPT(GPTConfig(**model_args))

    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    @torch.no_grad()
    def estimate_loss():
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch()
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        return losses.mean()
    
    trial_losses = []
    for trial in range(N):
        loss = estimate_loss().item()
        trial_losses.append(loss)
        print(f"Trial {trial+1}: {loss:.4f}")
    
    avg_loss = np.mean(trial_losses)
    std_loss = np.std(trial_losses)
    
    print(f"Average test loss (bpc) from {N} trials: {avg_loss:.4f} ± {std_loss:.4f}\n")