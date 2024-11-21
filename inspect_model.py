import torch
import os
import argparse
from model import GPTConfig, GPT
from adaptive_attention_span import AdaptiveGPT

def load_and_inspect_model(checkpoint_path, is_adaptive=False):
    # Load the checkpoint
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Get model arguments from checkpoint
    model_args = checkpoint['model_args']
    config = checkpoint.get('config', {})
    
    # Print basic model configuration
    print("\nModel Configuration:")
    for k, v in model_args.items():
        print(f"{k}: {v}")
    
    # Create model instance
    gptconf = GPTConfig(**model_args)
        
    if is_adaptive:
        print("\nDetected Adaptive Attention model")
        model = AdaptiveGPT(gptconf)
    else:
        print("\nDetected standard GPT model")
        model = GPT(gptconf)
    
    # Load state dict
    # first fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    model.load_state_dict(state_dict)
    
    # Print model statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Print training progress information
    print("\nTraining Progress:")
    print(f"Iteration number: {checkpoint['iter_num']}")
    print(f"Best validation loss: {checkpoint['best_val_loss']:.4f}")

    for index_block, block in enumerate(model.transformer.h):
        attn = block.attn
        print(f"\nBlock {index_block} Attention Span {index_block} - {attn.get_attention_span()}")
        #  print(f"Block {index_block} Attention Stride {index_block} - {attn.stride_params}")
    
    return model

if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description='Inspect a trained GPT model checkpoint')
    parser.add_argument('--checkpoint', type=str, default='out/ckpt.pt',
                        help='Path to the model checkpoint file')
    parser.add_argument('--adaptive', action='store_true',
                        help='Whether to inspect an adaptive attention model')
    args = parser.parse_args()
    
    if os.path.exists(args.checkpoint):
        model = load_and_inspect_model(args.checkpoint, is_adaptive=args.adaptive)
    else:
        print(f"No checkpoint found at {args.checkpoint}")
