out_dir = 'enwik8_6_babyPro_ext1x2_forgetting_and_bLR0'
eval_interval = 500
eval_iters = 200
log_interval = 100

always_save_checkpoint = False

wandb_log = True
wandb_project = 'enwik8'
wandb_run_name = 'enwik8_6_babyPro_ext1x2_forgetting_and_bLR0'

dataset = 'enwik8'
gradient_accumulation_steps = 2
batch_size = 32
block_size = 512

# half baby half gpt2 124M
n_layer = 8
n_head = 8
n_embd = 448
dropout = 0.1

learning_rate = 6e-4    # with baby networks can afford to go a bit higher
max_iters = 20000
lr_decay_iters = 20000  # make equal to max_iters usually
min_lr = 6e-5           # learning_rate / 10 usually
beta2 = 0.99            # make a bit bigger because number of tokens per iter is small

warmup_iters = 100      # not super necessary potentially

init_from = 'scratch'


is_forgetting = True
forget_interval = 2000
forget_strength = 0.64


is_bLR = True
circadian_amp = 0.3          # amplitude of fast fluctuations
circadian_period = 2000      # in iterations

long_cycle_amp = 0.2         # amplitude of slower cycles
long_cycle_period = 4000     # in iterations

noise_amp = 0.04             # optional small jitter

max_lr = 1.5e-3


is_attStr = False
attention_strides = [1] * 3 + [2] * 2  + [4] * 2 + [16] * 1     # total n_head elements