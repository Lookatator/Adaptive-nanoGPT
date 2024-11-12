# train on enwik8 dataset

out_dir = 'out-enwik8-char'
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
n_head = 2
n_embd = 512
dropout = 0.2
flash = True

learning_rate = 1e-3
max_iters = 100000
lr_decay_iters = 100000 # make equal to max_iters usually
min_lr = 3e-4 # learning_rate / 10 usually

warmup_iters = 100 # not super necessary potentially

# use Adaptive Attention Span
use_adaptive_attention = False

# wandb logging
wandb_log = True # override via command line if you like
wandb_project = 'enwik8-char'
if use_adaptive_attention:
    wandb_run_name = 'adaptive nano-gpt'
else:
    wandb_run_name = 'nano-gpt'
