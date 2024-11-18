import numpy as np
from model import GPT, MLP, AddLearnedPositionalEmbedding, GPTConfig, LayerNorm
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TriangleWave(nn.Module):
    """
    Implements a triangle wave with learnable period and offset_period_ratio using sigmoid interpolation
    f(x) = 4A/p * |((x - p/4) % p) - p/2| - A
    where A is amplitude, p is period (learned), and offset = period * offset_period_ratio (learned)
    offset_period_ratio is bounded between -1/4 and 1/4
    """

    def __init__(self,
                 period_min=2.,
                 period_max=4.0,
                 num_waves=1):
        super().__init__()

        # Store min/max bounds
        self.period_min = period_min
        self.period_max = period_max
        # Fixed bounds for offset_period_ratio
        self.ratio_min = -0.25  # -1/4
        self.ratio_max = 0.25  # 1/4

        # Initialize learnable weights for multiple waves
        init_period = torch.rand((num_waves,)) * (period_max - period_min) + period_min
        init_ratios = torch.zeros((num_waves,))  # Fixed initial ratio

        # Convert initial values to weight space using inverse sigmoid
        period_weight = self._inverse_interpolate(
            init_period, period_min, period_max)
        ratio_weight = self._inverse_interpolate(
            init_ratios, self.ratio_min, self.ratio_max)

        # Create learnable parameters with specified number of waves
        self.period_weight = nn.Parameter(period_weight)
        self.ratio_weight = nn.Parameter(ratio_weight)

    def _inverse_interpolate(self, value, min_val, max_val):
        """Convert a value in real space to weight space using inverse sigmoid"""
        normalized = (value - min_val) / (max_val - min_val)
        return torch.logit(normalized)

    def get_wave_params(self):
        """Get the actual period and offset_period_ratio values interpolated from weights"""
        period = self.period_min + (self.period_max - self.period_min) * torch.sigmoid(self.period_weight)
        ratio = self.ratio_min + (self.ratio_max - self.ratio_min) * torch.sigmoid(self.ratio_weight)
        return period, ratio

    def forward(self, mask):  # mask shape: (nh, T, T)
        # Get current period and offset_period_ratio
        period, ratio = self.get_wave_params()  # (nh,)

        # Calculate amplitude and offset based on period
        amplitude = period / 4.  # (nh,)
        offset = period * ratio  # (nh,)

        # Compute triangle wave
        # shifted_x = torch.fmod(mask, period.view(-1, 1, 1))  # (nh, T, T)
        # wave = 4 * (torch.abs(shifted_x - period.view(-1, 1, 1) / 2)) * amplitude.view(-1, 1, 1) / period.view(-1, 1, 1) - amplitude.view(-1, 1, 1) + 0.5 + offset.view(-1, 1, 1)  # (nh, T, T)
        wave = 0.5 * (torch.cos(2 * math.pi * mask / period.view(-1, 1, 1)) + 1) * amplitude.view(-1, 1, 1) + 0.5 + offset.view(-1, 1, 1)
        
        if torch.any(torch.isnan(wave)):
            print(f"Warning: NaN values in triangle wave for period {period} and ratio {ratio}")
            print(f"Wave: {wave}")
            print(f"Mask: {mask}")
            exit()

        # Calculate loss terms
        loss_terms = (1. / period) + (2. * ratio) - 1.0 / self.period_max - 2. * self.ratio_min

        # Clamp output to [0, 1]
        return torch.clamp(wave, min=0., max=1.), loss_terms

class AdaptiveCausalAttention(nn.Module):
    def get_attention_span(self):
        return torch.clamp((self.R + self.span_params) / self.R, min=0.0, max=1.0)

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        # key, query, value projections
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        
        # adaptive span parameters
        self.R = 32  # softness parameter for the masking function
        self.max_span = config.block_size  # maximum possible span
        self.span_params = nn.Parameter(torch.zeros(config.n_head))  # learnable span parameters
        self.span_reg = 2e-6  # L1 regularization coefficient for span parameters
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                            .view(1, 1, config.block_size, config.block_size))
        
        # Pre-compute position indices for masking
        positions = torch.arange(config.block_size, dtype=torch.float32)
        relative_pos = positions[:, None] - positions[None, :]  # (block_size, block_size)
        
        # Pre-compute base causal mask
        self.register_buffer('base_relative_pos', 
                             relative_pos)
        
        self.triangle_wave = TriangleWave(period_min=2., period_max=4., num_waves=self.n_head)

    def get_mask(self, span_param, max_length, device=None):
        """
        Create a shifting causal mask with adaptive span and stride.
        """
        T = max_length
        
        # Use pre-computed relative positions, sliced to current sequence length
        # Apply soft masking function - Use pre-computed relative positions, sliced to current sequence length
        relative_pos = self.base_relative_pos.masked_fill(self.base_relative_pos < 0, float('inf'))[:T,:T]
        relative_pos = torch.repeat_interleave(relative_pos.unsqueeze(0), span_param.shape[0], dim=0)  # (nh, T, T)
        mask_pos = torch.clamp((self.R - relative_pos + span_param.view(-1, 1, 1)) / self.R, min=0.0, max=1.0)  # (nh, T, T)
        
        # Apply triangle wave mask
        relative_pos = torch.clamp(self.base_relative_pos, min=0.0,)
        triangle_wave_mask, loss_terms_triangle_wave = self.triangle_wave(relative_pos)  #
        if torch.any(torch.isnan(triangle_wave_mask)):
            print("Warning: NaN values in triangle wave mask")
        
        # Combine mask_pos and triangle_wave_mask
        mask = mask_pos * triangle_wave_mask
        return mask, loss_terms_triangle_wave


    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality

        # Calculate query, key, values for all heads in batch
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # Calculate attention scores
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))  # (B, nh, T, T)
        
        # Clamp span parameters between 0 and max_span using sigmoid
        clamped_spans = torch.sigmoid(self.span_params) * self.max_span
        
        # Generate and apply adaptive attention masks with stride weights
        mask, loss_terms_triangle_wave = self.get_mask(clamped_spans, T, device=x.device)
        
        # Expand mask for batch dimension
        mask = mask.unsqueeze(0).expand(B, -1, -1, -1)  # (B, nh, T, T)
        
        # Apply mask
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        # Numerical stability: subtract max value before exp (standard log-sum-exp trick)
        att_max = torch.max(att, dim=-1, keepdim=True)[0]
        att_exp = torch.exp(att - att_max)
        
        # Apply mask after exp
        weighted_exp_att = att_exp * mask
        att = weighted_exp_att / weighted_exp_att.sum(dim=-1, keepdim=True)
        
        # Apply dropout
        att = self.attn_dropout(att)
        
        # Apply attention to values
        y = att @ v  # (B, nh, T, hs)
        
        # Restore time as batch dimension and concat heads
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # Output projection
        y = self.resid_dropout(self.c_proj(y))

        if torch.any(torch.isnan(y)):
            print("Warning: NaN values in output")
            print(torch.max(att))
            print(att)
            print(mask)
            print(weighted_exp_att)
            exit()
        
        # Calculate combined regularization loss using pre-calculated weights
        strides_loss = loss_terms_triangle_wave.view(-1, 1)  # (nh, 1)
        loss_per_head = (clamped_spans + self.R) * strides_loss.squeeze(-1)  # (nh,)
        # loss_per_head = clamped_spans  # (nh,)

        span_loss = self.span_reg * loss_per_head.sum() / self.n_head
        if torch.any(torch.isnan(span_loss)):
            print("Warning: NaN values in span loss")

        # Collect extra info about spans and losses
        extra_info = {
            'spans': clamped_spans.detach().cpu().numpy(),  # Shape: (n_head,)
            'triangle_wave_losses': loss_terms_triangle_wave.detach().cpu().numpy(),  # Shape: (n_head,)
        }

        return y, span_loss, extra_info

class AdaptiveBlock(nn.Module):
    def __init__(self, config, add_extra_pos_emb):
        super().__init__()
        if add_extra_pos_emb:
            self.add_extra_pos_emb = AddLearnedPositionalEmbedding(config)
        else:
            self.add_extra_pos_emb = None
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = AdaptiveCausalAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        if self.add_extra_pos_emb is not None:
            x = self.add_extra_pos_emb(x)
        attn_out, span_loss, extra_info_attn = self.attn(self.ln_1(x))
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, span_loss, extra_info_attn

class AdaptiveGPT(GPT):
    def __init__(self, config):
        super().__init__(config)
        blocks = []
        for index_layer in range(config.n_layer):
            if index_layer == 0:
                print(f"Block {index_layer} does not add positional embeddings")
                add_extra_pos_emb = False
            else:
                print(f"Block {index_layer} adds positional embeddings")
                add_extra_pos_emb = True
            blocks.append(AdaptiveBlock(config, add_extra_pos_emb))
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList(blocks),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("AdaptiveGPT number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def forward(self, idx, targets=None, add_span_loss=True):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        # Forward pass through the model
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        
        # Accumulate span losses from all blocks
        total_span_loss = 0
        extra_info_list = []
        for block in self.transformer.h:
            x, span_loss, extra_info_block = block(x)
            total_span_loss += span_loss
            extra_info_list.append(extra_info_block)

        # Concatenate all dictionaries from extra_info_block
        combined_extra_info = {}
        for block_info in extra_info_list:
            for key, value in block_info.items():
                if key not in combined_extra_info:
                    combined_extra_info[key] = value
                else:
                    combined_extra_info[key] = np.concatenate([combined_extra_info[key], value])
        extra_info = combined_extra_info

        x = self.transformer.ln_f(x)

        if targets is not None:
            # Calculate loss including the span regularization
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            if add_span_loss:
                loss = loss + total_span_loss
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss, extra_info

if __name__ == "__main__":
    # test get mask
    causal_attn = AdaptiveCausalAttention(GPTConfig())
    mask, one_hot = causal_attn.get_mask(0.5, 1024, stride_weights=torch.tensor([0.5, 0.5]))
    print(mask)
    print(one_hot)
