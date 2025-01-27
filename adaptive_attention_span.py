import numpy as np
from model import GPT, MLP, AddLearnedPositionalEmbedding, GPTConfig, LayerNorm
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TriangleWave(nn.Module):
    """
    Implements a triangle wave with learnable period and offset_period_ratio using sigmoid interpolation
    the triangle wave is approximated using a Fourier series. The amplitude is equal to period / 4 (half the peak-to-peak distance)
    and the offset is the period * offset_period_ratio (learned).
    """
    def approximate_triangle_wave(self, x, period, amplitude, max_harmonics=10):
        """
        Approximate a triangle wave using a Fourier series
        :param x: Input tensor
        :param period: Period of the triangle wave
        :param amplitude: Amplitude of the triangle wave
        :return: Approximated triangle wave
        """
        # Compute Fourier series coefficients
        k = torch.arange(1, max_harmonics + 1, 1, dtype=x.dtype, device=x.device)
        coeff = 8.0 * (1.0 - (-1.0) ** k) / ((math.pi ** 2) * (k ** 2))
        # Compute Fourier series, expand dimensions for broadcasting
        x = x.unsqueeze(-1).expand(*x.size(), k.size(0))  # (nh, T, T, num_harmonics)
        k = k.view(1, 1, 1, -1)  # (1, 1, 1, num_harmonics)
        period = period.view(-1, 1, 1, 1)  # (nh, 1, 1, 1)
        coeff = coeff.view(1, 1, 1, -1)  # (1, 1, 1, num_harmonics)
        phase = 2.0 * math.pi * x / period
        phase = torch.fmod(phase, 2.0 * math.pi)
        wave = torch.sum(torch.cos(phase * k) * coeff, dim=3) # (nh, T, T)
        # Scale and shift the wave
        amplitude = amplitude.view(-1, 1, 1)  # (nh, 1, 1)
        wave = wave * amplitude / 2  # (nh, T, T)
        return wave


    def __init__(self,
                 period_min=2.,
                 period_max=8.0,
                 num_waves=1,
                 max_harmonics=5):
        """
        :param period_min: Minimum period of the triangle wave
        :param period_max: Maximum period of the triangle wave
        :param num_waves: Number of waves in the triangle wave (equal to number of heads in the attention layer)
        :param max_harmonics: Maximum number of harmonics to use in the Fourier series approximation of the triangle wave
        """
        super().__init__()

        # Store min/max bounds
        self.period_min = period_min
        self.period_max = period_max
        # Fixed bounds for offset_period_ratio
        self.ratio_min = -0.25  # -1/4
        self.ratio_max = 0.25  # 1/4

        self.max_harmonics = max_harmonics

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
        wave = self.approximate_triangle_wave(mask, period, amplitude, max_harmonics=self.max_harmonics)
        wave = wave + 0.5 + offset[..., None, None]

        # Calculate loss terms
        base_terms = (1. / period) + (2. * ratio) + 0.5
        loss_terms = torch.where(
            base_terms < 1.0,
            base_terms,
            1.0 + (0.5 + offset - amplitude),
        )

        # Clamp output to [0, 1]
        return torch.clamp(wave, min=0., max=1.), loss_terms

class AdaptiveCausalAttention(nn.Module):
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
        self.R = config.softness_span_mask  # softness parameter for the masking function
        self.span_reg = config.span_reg  # L1 regularization coefficient for span parameters
        self.max_span = config.block_size  # maximum possible span
        
        self.span_params = nn.Parameter(torch.zeros(config.n_head))  # learnable span parameters
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                            .view(1, 1, config.block_size, config.block_size))
        
        # Pre-compute position indices for masking
        positions = torch.arange(config.block_size, dtype=torch.float32)
        relative_pos = positions[:, None] - positions[None, :]  # (block_size, block_size)
        
        # Pre-compute base causal mask
        self.register_buffer('base_relative_pos', 
                             relative_pos)
        
        self.use_triangle_wave_masking = config.use_triangle_wave_masking
        
        # Only create triangle wave masking if we're using it
        if self.use_triangle_wave_masking:
            self.triangle_wave = TriangleWave(period_min=config.period_min_triangle_wave_masking, 
                                              period_max=config.period_max_triangle_wave_masking, 
                                              num_waves=self.n_head)
        else:
            self.triangle_wave = None

    def get_attention_span(self):
        clamped_spans = torch.sigmoid(self.span_params) * self.max_span + self.R
        return torch.floor(torch.clamp(clamped_spans, min=0.0, max=self.max_span))
    
    def get_triangle_wave_mask(self):
        if not self.use_triangle_wave_masking:
            return None
            
        # Generate positions from 0 to max_length-1
        positions = torch.arange(self.max_span, dtype=torch.float32, device=self.span_params.device)
        
        # Apply triangle wave to positions
        triangle_mask, _ = self.triangle_wave(positions.unsqueeze(0).unsqueeze(-1))

        return triangle_mask.squeeze(0)
    
    def get_non_masked_tokens(self):
        if not self.use_triangle_wave_masking:
            return None
        
        attention_span = self.get_attention_span()
        positions = torch.arange(self.max_span, dtype=torch.float32, device=self.span_params.device)
        positions = positions.unsqueeze(0).expand(attention_span.shape[0], -1)
        attention_span = attention_span.unsqueeze(-1).expand(-1, positions.shape[1])
        masked_positions = positions.clone()
        masked_positions[positions > attention_span] = 0.0
        masked_positions[positions <= attention_span] = 1.0

        # count number of zeros in mask
        return self.get_triangle_wave_mask().squeeze(-1) * masked_positions 

    def get_span_mask(self, span_param, max_length):
        """
        Create a shifting causal mask with adaptive span.
        """
        T = max_length
        
        # Use pre-computed relative positions, sliced to current sequence length
        relative_pos = self.base_relative_pos.masked_fill(self.base_relative_pos < 0, float('inf'))[:T,:T]
        relative_pos = relative_pos.unsqueeze(0).expand(span_param.shape[0], -1, -1)  # (nh, T, T)
        mask_pos = torch.clamp((self.R - relative_pos + span_param.view(-1, 1, 1)) / self.R, min=0.0, max=1.0)  # (nh, T, T)
        
        return mask_pos

    def get_triangle_mask(self, span_param, max_length):
        """
        Create a triangle wave mask if enabled, otherwise return default mask.
        """
        T = max_length
        
        if self.use_triangle_wave_masking:
            # Apply triangle wave masking
            relative_pos = torch.clamp(self.base_relative_pos, min=0.0,)[:T,:T]
            relative_pos = relative_pos.unsqueeze(0).expand(span_param.shape[0], -1, -1)  # (nh, T, T)
            triangle_wave_mask, loss_terms_triangle_wave_masking = self.triangle_wave(relative_pos,)
            if torch.any(torch.isnan(triangle_wave_mask)):
                print("Warning: NaN values in triangle wave masking")
        else:
            # If not using triangle wave masking, return ones for the mask and zeros for loss terms
            triangle_wave_mask = torch.ones((span_param.shape[0], T, T))
            loss_terms_triangle_wave_masking = None
        
        return triangle_wave_mask, loss_terms_triangle_wave_masking


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
        mask_span = self.get_span_mask(clamped_spans, T)
        
        # Expand mask for batch dimension
        mask_span = mask_span.unsqueeze(0).expand(B, -1, -1, -1)  # (B, nh, T, T)
        log_mask_span = torch.log(mask_span.clamp(min=1e-6))
        
        if self.use_triangle_wave_masking:
            mask_triangle_wave, loss_terms_triangle_wave_masking = self.get_triangle_mask(clamped_spans, T)
            mask_triangle_wave = mask_triangle_wave.unsqueeze(0).expand(B, -1, -1, -1)  # (B, nh, T, T)
            log_mask_triangle_wave_masking = torch.log(mask_triangle_wave.clamp(min=1e-6))
            weighted_att = att + log_mask_span + log_mask_triangle_wave_masking
            weighted_att = weighted_att.masked_fill(torch.minimum(mask_span, mask_triangle_wave) <= 1e-6, float('-inf'))
        else:
            weighted_att = att + log_mask_span
            weighted_att = weighted_att.masked_fill(mask_span <= 1e-6, float('-inf'))

        # calculate values
        att = F.softmax(weighted_att, dim=-1)
        att_dropout = self.attn_dropout(att)
        y = att_dropout @ v  # (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        
        # Loss
        clamped_spans_with_R = clamped_spans + self.R
        if self.use_triangle_wave_masking:
            # Calculate combined regularization loss using pre-calculated weights
            strides_loss = loss_terms_triangle_wave_masking.view(-1, 1)  # (nh, 1)
            loss_per_head = clamped_spans_with_R * strides_loss.squeeze(-1)  # (nh,)
        else:
            # if not using triangle wave masking, only use span loss
            loss_per_head = clamped_spans

        adaptive_span_regularization_loss = self.span_reg * loss_per_head.sum() / self.n_head

        # Collect extra info about spans and losses
        extra_info = {
            'clamped_spans': clamped_spans_with_R.detach().cpu().numpy(),  # Shape: (n_head,)
            'adaptive_span_loss': adaptive_span_regularization_loss.detach().cpu().numpy().reshape(1,),  # Shape: (1,)
            'attention_weights': weighted_att[:, :, -1, :].mean(dim=0).detach().cpu().numpy()  # Shape: (nh, T)
        }

        if self.use_triangle_wave_masking:
            periods, ratios = self.triangle_wave.get_wave_params()
            offsets = periods * ratios
            extra_info['triangle_wave_periods'] = periods.detach().cpu().numpy()
            extra_info['triangle_wave_offsets'] = offsets.detach().cpu().numpy()
            extra_info['triangle_wave_losses'] = loss_terms_triangle_wave_masking.detach().cpu().numpy()

        return y, adaptive_span_regularization_loss, extra_info

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
