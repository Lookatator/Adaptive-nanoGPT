import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

def create_layer_head_heatmaps(attention_matrices, save_dir=None):
    """
    Creates heatmaps for each layer and head combination.
    
    Args:
        attention_matrices: numpy array of shape (n_layers, n_heads, seq_len, seq_len)
        save_dir: optional directory to save individual heatmaps
    """
    n_layers, n_heads = attention_matrices.shape[:2]

    # Reshape attention matrices to create layer-head grid
    n_total = n_layers * n_heads
    attention_matrices = attention_matrices.reshape(n_total, -1)
    # attention_matrices = attention_matrices / attention_matrices.sum(axis=-1, keepdims=True)

    
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    # attention_matrices = attention_matrices / attention_matrices.sum(axis=-1, keepdims=True)
    
    # Create figure and axis
    plt.figure(figsize=(12, 8))
    
    # Create heatmap
    sns.heatmap(attention_matrices, cmap='Blues', cbar_kws={'label': 'Attention Weight'},)
    
    # Customize axis labels
    plt.xlabel('Token Position')
    plt.ylabel('Index Head')

    # Add layer-head labels on y-axis
    layer_head_labels = [f'L{index_layer+1}H{h+1}' if h == 0 else '' for index_layer in range(n_layers) for h in range(n_heads)]
    plt.yticks(np.arange(n_total) + 0.5, layer_head_labels, rotation=0)
    xlabels = [f'{i}' if i % 10 == 0 else '' for i in np.arange(attention_matrices.shape[-1])[::-1]]
    plt.xticks(np.arange(attention_matrices.shape[-1]) + 0.5, xlabels, rotation=90)
    
    plt.title('Attention Patterns Across All Layers and Heads')
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(save_dir / 'attention_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

# Example usage
if __name__ == "__main__":
    import argparse
    import torch
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--att', type=str, required=True, help='Path to attention matrices (.npy file)')
    parser.add_argument('--n_layers', type=int, required=True, help='Number of layers')
    parser.add_argument('--n_heads', type=int, required=True, help='Number of heads')
    args = parser.parse_args()

    # Load attention matrices
    attention_matrices = torch.load(args.att)
    attention_matrices = attention_matrices.reshape(args.n_layers, args.n_heads, -1)
    attention_matrices = torch.from_numpy(attention_matrices).float()
    attention_matrices = torch.softmax(attention_matrices, dim=-1).detach().cpu().numpy()
    attention_matrices = attention_matrices / (attention_matrices.max(axis=-1, keepdims=True) + 1e-6)
    # attention_matrices[~np.isinf(attention_matrices)] = 1.
    # attention_matrices[np.isinf(attention_matrices)] = -30
    # attention_matrices[attention_matrices > -30.] = attention_matrices[attention_matrices > -30.] + 1

    print(attention_matrices.shape)


    # Create a heatmap for the average attention across all layers and heads
    create_layer_head_heatmaps(attention_matrices, save_dir="average_attention_heatmap")