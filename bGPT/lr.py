import math
import numpy as np

def get_blr(iter_num, config, rng=None):
    base_lr = config["learning_rate"]

    if config["decay_lr"]:
        if iter_num < config["warmup_iters"]:
            decay_mult = iter_num / config["warmup_iters"]
        else:
            decay_progress = min(1.0, (iter_num - config["warmup_iters"]) / (config["lr_decay_iters"] - config["warmup_iters"]))
            decay_mult = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
        decayed_lr = base_lr * decay_mult
    else:
        decayed_lr = base_lr

    circadian = config["circadian_amp"] * math.cos(2 * math.pi * iter_num / config["circadian_period"])
    long_cycle = config["long_cycle_amp"] * math.cos(2 * math.pi * iter_num / config["long_cycle_period"])

    if rng is not None:
        noise = rng.normal(loc=0.0, scale=config["noise_amp"]) if config["noise_amp"] > 0 else 0.0
    else:
        noise = np.random.normal(loc=0.0, scale=config["noise_amp"]) if config["noise_amp"] > 0 else 0.0

    efficiency = 1.0 + circadian + long_cycle + noise
    lr = decayed_lr * efficiency
    lr = max(config["min_lr"], min(config["max_lr"], lr))

    return lr
