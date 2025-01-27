from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from analysis.load_model import load_model



def create_comparison_plot(models_data, save_path=None):
    # Configuration
    n_layers = 12
    n_heads = 8
    # Updated to create 3 subplots with adjusted height ratios
    fig, (ax1, ax_span, ax2) = plt.subplots(
        3, 1, figsize=(12, 12), height_ratios=[1, 1, 1]
    )

    # Color palette for different models
    colors = ["#2ecc71", "#3498db", "#e74c3c"]
    labels = ["nanoGPT Baseline", "Adaptive Span", "Adaptive Span + Periodic Masking"]

    # X-axis positions
    x = np.arange(n_layers * n_heads)

    # Top plot: Non-masked tokens per head
    lines = []
    for i, (model_name, data) in enumerate(models_data.items()):
        non_masked = data["non_masked_tokens"]
        # Reshape to (n_layers, n_heads) for layer-wise sorting
        non_masked_reshaped = non_masked.reshape(n_layers, n_heads)
        triangle_cont = data.get("triangle_contribution", np.zeros_like(non_masked))
        triangle_cont_reshaped = triangle_cont.reshape(n_layers, n_heads)
        attention_span = data.get("attention_span", np.zeros_like(non_masked))
        attention_span_reshaped = attention_span.reshape(n_layers, n_heads)

        # Sort each layer independently
        for layer in range(n_layers):
            sort_indices = np.argsort(non_masked_reshaped[layer])
            non_masked_reshaped[layer] = non_masked_reshaped[layer][sort_indices]
            triangle_cont_reshaped[layer] = triangle_cont_reshaped[layer][sort_indices]
            attention_span_reshaped[layer] = attention_span_reshaped[layer][
                sort_indices
            ]

        # Flatten back to original shape
        non_masked = non_masked_reshaped.flatten()
        data["triangle_contribution"] = triangle_cont_reshaped.flatten()
        data["attention_span"] = attention_span_reshaped.flatten()

        # Plot for first subplot (non-masked tokens)
        line = []
        for layer in range(n_layers):
            start_idx = layer * n_heads
            end_idx = (layer + 1) * n_heads
            layer_line = ax1.plot(
                x[start_idx:end_idx],
                non_masked[start_idx:end_idx],
                "o-",
                color=colors[i],
                alpha=0.7,
            )[0]
            line.append(layer_line)
        line = line[0]  # Use first line for legend
        lines.append(line)

        # Plot for middle subplot (attention span)
        ax_span.plot(
            x, data.get("attention_span", non_masked), "o", color=colors[i], alpha=0.7
        )

        # Plot for bottom subplot (triangle wave contribution)
        if "triangle_contribution" in data:
            triangle_cont = data["triangle_contribution"]
            ax2.plot(x, triangle_cont * 100, "o", color=colors[i], alpha=0.7)

    # Add vertical lines and grid to all plots
    for ax in [ax1, ax_span, ax2]:
        for layer in range(1, n_layers):
            ax.axvline(x=layer * n_heads - 0.5, color="gray", linestyle="--", alpha=0.3)
        ax.grid(True, alpha=0.3)

    ax1.set_ylabel("Number of Non-masked Tokens")
    ax_span.set_ylabel("Attention Span")
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Periodic Masking Contribution (%)")

    # Set y-axis to log scale for ax1 and ax_span
    ax1.set_yscale('log')
    ax_span.set_yscale('log')

    # Set minimum y-axis limit to 10 for log-scaled plots
    ax1.set_ylim(bottom=10)
    ax_span.set_ylim(bottom=10)
    ax2.set_ylim(bottom=-5, top=105)


    # X-axis labels for all plots
    for ax in [ax1, ax_span, ax2]:
        for layer in range(n_layers):
            x_pos = layer * n_heads + (n_heads / 2 - 0.5)
            y_pos = ax.get_ylim()[0] - (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.05
            if ax == ax2:
                ax.text(x_pos, y_pos, f"Layer {layer+1}", ha="center", va="top")
        ax.set_xticks([])
        ax.set_xlabel("")

    # Single legend at the bottom
    fig.legend(lines, labels, loc="center", bbox_to_anchor=(0.5, 0), ncol=3)

    plt.tight_layout()
    # Adjust layout to make room for legend
    plt.subplots_adjust(bottom=0.1)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, (ax1, ax_span, ax2)


def collect_attention_spans(model):
    attention_spans = []
    for block in model.transformer.h:
        attn = block.attn
        attention_spans.append(attn.get_attention_span())
    return torch.cat(attention_spans).detach().cpu().numpy()

def collect_triangle_wave_mask_rate(model):
    triangle_wave_mask_rate = []
    for block in model.transformer.h:
        attn = block.attn
        mask = attn.get_triangle_wave_mask()
        if mask is not None:
            # calculate number of zeros in mask
            zero_count = (mask <= 0.0).float().sum(axis=1)
            triangle_wave_mask_rate.append(zero_count / mask.shape[1])
        
    return torch.cat(triangle_wave_mask_rate).detach().cpu().numpy()

def collect_non_masked_tokens(model):
    non_masked_tokens = []
    for block in model.transformer.h:
        attn = block.attn
        mask = attn.get_non_masked_tokens()
        if mask is not None:
            non_zero_count = torch.cumsum((mask > 0.0).float(), dim=1).sum(axis=1)
            non_masked_tokens.append(non_zero_count)
    return torch.cat(non_masked_tokens).detach().cpu().numpy()

# Example usage
if __name__ == "__main__":
    path_results = "results/problem_set"
    path_save = Path(path_results)
    path_model_baseline = Path(path_results) / "out_20241121_012958_baseline" / "ckpt.pt"
    path_model_adaptive_span = Path(path_results) / "out_20241122_012032_adaptive_span" / "ckpt.pt"
    path_model_adaptive_span_periodic = Path(path_results) / "out_20241121_032355_adaptive_span_periodic" / "ckpt.pt"

    model_baseline = load_model(path_model_baseline, is_adaptive=False)
    model_adaptive_span = load_model(path_model_adaptive_span, is_adaptive=True)
    model_adaptive_span_periodic = load_model(path_model_adaptive_span_periodic, is_adaptive=True)



    # for index_block, block in enumerate(model_adaptive_span.transformer.h):
    #     attn = block.attn
    #     print(f"\nBlock {index_block} Attention Span {index_block} - {attn.get_attention_span()}")
    #     #  print(f"Block {index_block} Attention Stride {index_block} - {attn.stride_params}")

    block_size = model_baseline.config.block_size

    # attention_spans_baseline = collect_attention_spans(model_baseline)
    attention_spans_adaptive_span = collect_attention_spans(model_adaptive_span)
    attention_spans_adaptive_span_periodic = collect_attention_spans(model_adaptive_span_periodic)
    triangle_wave_mask_rate_adaptive_span_periodic = collect_triangle_wave_mask_rate(model_adaptive_span_periodic)
    non_masked_tokens_adaptive_span_periodic = collect_non_masked_tokens(model_adaptive_span_periodic)
    # Example data structure (replace with your actual data)
    n_heads = 8
    n_layers = 12
    models_data = {
        "nanogpt": {
            "non_masked_tokens": np.full(n_layers * n_heads, block_size),
            # 'triangle_contribution': np.zeros(n_layers * n_heads),
            "attention_span": np.full(n_layers * n_heads, block_size),
        },
        "adaptive_span": {
            "non_masked_tokens": attention_spans_adaptive_span.copy(),
            # 'triangle_contribution': np.zeros(n_layers * n_heads),
            "attention_span": attention_spans_adaptive_span.copy(),  # Added attention span data
        },
        "adaptive_span_periodic": {
            "non_masked_tokens": non_masked_tokens_adaptive_span_periodic.copy(),
            "triangle_contribution": triangle_wave_mask_rate_adaptive_span_periodic.copy(),
            "attention_span": attention_spans_adaptive_span_periodic.copy(),  # Added attention span data
        },
    }

    fig, axes = create_comparison_plot(models_data, save_path="masking_analysis.png")
    plt.savefig("masking_analysis.png", dpi=300, bbox_inches="tight")

    total_non_masked_tokens_span_periodic = np.sum(non_masked_tokens_adaptive_span_periodic)
    print(f"Total non-masked entries with periodic masking: {total_non_masked_tokens_span_periodic}")


    total_non_masked_tokens_span = np.sum(
        block_size * (block_size + 1) / 2 - (block_size - attention_spans_adaptive_span) * (block_size - attention_spans_adaptive_span + 1) / 2
    )
    print(f"Total non-masked entries with adaptive span: {total_non_masked_tokens_span}")
