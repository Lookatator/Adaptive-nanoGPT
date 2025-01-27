import torch
import os
import argparse
from model import GPTConfig, GPT
from adaptive_attention_span import AdaptiveGPT


def load_model(checkpoint_path, is_adaptive=False):
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
    
    # Handle adaptive attention span parameters compatibility
    if 'period_min_triangle_wave' in model_args:
        model_args['period_min_triangle_wave_masking'] = model_args['period_min_triangle_wave']
        del model_args['period_min_triangle_wave']
    if 'period_max_triangle_wave' in model_args:
        model_args['period_max_triangle_wave_masking'] = model_args['period_max_triangle_wave']
        del model_args['period_max_triangle_wave']

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
    
    return model
