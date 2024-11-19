# train on enwik8 dataset

import time


eval_interval = 1000
eval_iters = 200
log_interval = 1000 # don't print too too often

always_save_checkpoint = True

dataset = 'enwik8'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 512 # context of up to 512 previous characters

# baby GPT model :)
n_layer = 12
n_head = 8
n_embd = 512
dropout = 0.2
flash = True

max_iters = 150000

# Learning rate
learning_rate = 1e-3
decay_lr = True
lr_decay_iters = 150000 # make equal to max_iters usually
min_lr = 1e-4 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 100 # not super necessary potentially

# use Adaptive Attention Span
use_adaptive_attention = True
softness_span_mask = 32
span_reg = 2e-6

# wandb logging
wandb_log = True # override via command line if you like
wandb_project = 'enwik8-char'
if use_adaptive_attention:
    wandb_run_name = 'adaptive nano-gpt'
else:
    wandb_run_name = 'nano-gpt'

out_dir = f'results/out_{time.strftime("%Y%m%d_%H%M%S")}_{"adaptive_span" if use_adaptive_attention else "causal"}'
