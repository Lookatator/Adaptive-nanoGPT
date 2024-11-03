# train on enwik8 dataset

out_dir = 'out-enwik8-char'
eval_interval = 1000
eval_iters = 200
log_interval = 1000 # don't print too too often

always_save_checkpoint = True

wandb_log = True # override via command line if you like
wandb_project = 'enwik8-char'
wandb_run_name = 'nano-gpt'

dataset = 'enwik8'
gradient_accumulation_steps = 1
batch_size = 32
block_size = 512 # context of up to 512 previous characters

# baby GPT model :)
n_layer = 12
n_head = 2
n_embd = 512
dropout = 0.2
flash = True

learning_rate = 6e-4
max_iters = 150000
lr_decay_iters = 50000 # make equal to max_iters usually
min_lr = 1e-4 # learning_rate / 10 usually
# beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 100 # not super necessary potentially

# on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model
