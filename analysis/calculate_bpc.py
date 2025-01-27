
import argparse
import math
import os
import numpy as np
import torch
import torch.nn.functional as F
from contextlib import nullcontext
from tqdm import tqdm

from analysis.load_model import load_model

# -----------------------------------------------------------------------------
# Device settings
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device == 'cpu' else torch.amp.autocast(device_type=device, dtype=ptdtype)


@torch.no_grad()
def get_bpc(model, data_dir, block_size, batch_size, device, file='test.bin', stride=None, is_adaptive=False):
    data = np.memmap(os.path.join(data_dir, file), dtype=np.uint16, mode='r')
    nlls = []
    prev_end_loc = 0
    seq_len = len(data)

    if stride is None:
        stride = block_size

    batch_size_eval = batch_size * 8
    all_attention_weights = 0.
    count_loops = 0

    for begin_loc in tqdm(range(0, seq_len, stride * batch_size_eval), desc='Calculating BPC'):
        end_locs = [min(begin_loc + block_size + i * stride, seq_len - 1) for i in range(batch_size_eval)]
        prev_end_locs = [prev_end_loc] + end_locs[:-1] # prev_end_loc is the end_loc of the previous batch

        # trg_lens is the length of the target sequence for each sequence. Should be the same as stride most of the time
        # except on the last loop, where it's different, on the first loop, it is equal to block_size, and on the last loop,
        # depends on the full seq_len.
        trg_lens = [end_locs[i] - prev_end_locs[i] for i in range(batch_size_eval)]  

        # We don't want to compute the loss for sequences that are 0 length.
        trg_lens = [trg_len for trg_len in trg_lens if trg_len > 0]


        if len(trg_lens) > 0:
            # Calculate starting positions for each sequence in the batch
            begin_locs = np.arange(0, batch_size_eval) * stride + begin_loc
            begin_locs = begin_locs[:len(trg_lens)]
            end_locs = end_locs[:len(trg_lens)]
            begin_locs = [min(begin_loc, seq_len - block_size - 1) for begin_loc in begin_locs]
            
            # Create input and target tensors
            x = torch.stack([torch.from_numpy((data[begin_locs[i]:end_locs[i]]).astype(np.int64)) for i in range(len(trg_lens))])
            y = torch.stack([torch.from_numpy((data[begin_locs[i]+1:end_locs[i]+1]).astype(np.int64)) for i in range(len(trg_lens))])

            if device == 'cuda':
                # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
                x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
            else:
                x, y = x.to(device), y.to(device)

            target_ids = y.clone()
            for i, trg_len in enumerate(trg_lens):
                target_ids[i, :-trg_len] = -1  # Mask tokens we don't want to predict

            with ctx:
                # Forward pass through model
                logits, _, extra_info = model(x, targets=y)
                if is_adaptive:
                    all_attention_weights += extra_info['attention_weights']
                count_loops += 1
                # Calculate loss, ignoring padding tokens (-1)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_ids.view(-1), ignore_index=-1)
                neg_log_likelihood = loss

            nlls.append(neg_log_likelihood)

        prev_end_loc = end_locs[-1]
        if end_locs[-1] == seq_len:
            break

    bpc = torch.stack(nlls).mean() / math.log(2) # bits per character (using log base 2)

    extra_info = {}

    if is_adaptive:
        average_attention_weights = all_attention_weights / count_loops
        extra_info['average_attention_weights'] = average_attention_weights

    return bpc, extra_info

def get_args():
    parser = argparse.ArgumentParser(description='Calculate bits per character (BPC) for a trained model')
    parser.add_argument('--checkpoint', type=str, default='out/ckpt.pt',
                        help='Path to the model checkpoint file')
    parser.add_argument('--data_dir', type=str, default='data/shakespeare',
                        help='Path to data directory containing train.bin and val.bin')
    parser.add_argument('--adaptive', action='store_true',
                        help='Whether to evaluate an adaptive attention model')
    parser.add_argument('--batch_size', type=int, default=12,
                        help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to run evaluation on')
    parser.add_argument('--stride', type=int, default=None,
                        help='Stride length for evaluation (defaults to block size if not specified)')
    return parser.parse_args()

def main():
    args = get_args()

    # Load the model
    model = load_model(args.checkpoint, is_adaptive=args.adaptive)
    model.to(args.device)
    model.eval()

    # Get block size from model config
    block_size = model.config.block_size
    
    # Use block_size as stride if not specified
    stride = args.stride if args.stride is not None else block_size

    # Calculate BPC
    with torch.no_grad():
        test_bpc, extra_info = get_bpc(model, args.data_dir, block_size, args.batch_size, args.device, stride=stride, is_adaptive=args.adaptive)
    
    print(f"Test BPC: {test_bpc:.4f}")
    if args.adaptive:
        print(f"Average attention weights: {extra_info['average_attention_weights']}")
        torch.save(extra_info['average_attention_weights'], 'average_attention_weights.pt')

if __name__ == "__main__":
    main()
