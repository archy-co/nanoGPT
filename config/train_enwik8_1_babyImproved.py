out_dir = 'enwik8_test1_babyImproved'
eval_interval = 500
eval_iters = 200
log_interval = 50

always_save_checkpoint = False

wandb_log = True
wandb_project = 'enwik8'
wandb_run_name = 'enwik8_1_babyImproved'

dataset = 'enwik8'
gradient_accumulation_steps = 2
batch_size = 32
block_size = 512

# gpt2 model 124M params
# n_layer = 12
# n_head = 12
# n_embd = 768
# dropout = 0.1

# baby GPT model :)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.1

learning_rate = 6e-4    # with baby networks can afford to go a bit higher
max_iters = 10000
lr_decay_iters = 10000  # make equal to max_iters usually
min_lr = 6e-5           # learning_rate / 10 usually
beta2 = 0.99            # make a bit bigger because number of tokens per iter is small

warmup_iters = 100      # not super necessary potentially
