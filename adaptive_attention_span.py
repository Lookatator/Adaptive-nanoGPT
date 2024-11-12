from model import GPT, MLP, AddLearnedPositionalEmbedding, GPTConfig, LayerNorm
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class AdaptiveCausalAttention(nn.Module):
    @classmethod
    def create_special_matrix(cls, size, S):
        """
        Creates a square matrix with:
        - 1's on the diagonal
        - 1's every S entries before the diagonal
        - 0's everywhere else
        
        Args:
            size (int): Size of the square matrix
            S (int): Spacing between 1's before the diagonal
            
        Returns:
            torch.Tensor: The resulting matrix
        """
        # Create indices for all positions in the matrix
        i, j = torch.meshgrid(torch.arange(size), torch.arange(size), indexing='ij')
        
        # Create the matrix using boolean operations:
        # 1. Diagonal: i == j
        # 2. Below diagonal (i > j) with spacing S: (i - j) % S == 0
        matrix = ((i == j) | ((i > j) & ((i - j) % S == 0))).float()
        
        return matrix

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
        
        # stride selection parameters
        self.strides = [1, 2]  # possible stride values
        for stride in self.strides:
            self.register_buffer(f"special_matrix_{stride}", 
                                 self.create_special_matrix(config.block_size, stride)
                                    .view(config.block_size, config.block_size))
        self.stride_params = nn.Parameter(torch.zeros(self.n_head, 2))  # learnable stride parameters
        self.stride_reg = 1e-6  # regularization coefficient for stride parameters
        

    def get_mask(self, span_param, max_length, device=None, stride_weights=None):
        """
        Create a shifting causal mask with adaptive span and stride.
        
        Args:
            span_param: float or tensor, the learned span parameter
            max_length: int, the sequence length
            device: torch.device, the device to create the mask on
            stride_weights: tensor of shape (2,), weights for stride 1 and 2
        """
        T = max_length
        # Create position indices matrix for both strides
        positions = torch.arange(max_length, device=device, dtype=torch.float32)
        
        # Create matrices of relative positions for both strides
        relative_pos_1 = positions[:, None] - positions[None, :]  # stride 1
        relative_pos_2 = positions[:, None] - positions[None, :]  # stride 2
        
        # Mask future positions for both strides
        relative_pos_1 = relative_pos_1.masked_fill(relative_pos_1 < 0, float('inf'))
        relative_pos_2 = relative_pos_2.masked_fill(relative_pos_2 < 0, float('inf'))

        relative_pos_1 = relative_pos_1.to(device)
        relative_pos_2 = relative_pos_2.to(device)

        relative_pos_1 = relative_pos_1.masked_fill(self.special_matrix_1[:T,:T] == 0, float('inf'))
        relative_pos_2 = relative_pos_2.masked_fill(self.special_matrix_2[:T,:T] == 0, float('inf'))
        
        # Apply soft masking function for both strides
        mask_1 = torch.clamp((self.R + span_param - relative_pos_1) / self.R, min=0.0, max=1.0)
        mask_2 = torch.clamp((self.R + span_param - relative_pos_2) / self.R, min=0.0, max=1.0)
        
        # Combine masks using stride weights
        # Sample mask choice using stride weights as categorical probabilities

        # choice = torch.multinomial(stride_weights, 1)[0]
        # # Create one-hot with straight-through gradient
        # one_hot = F.one_hot(choice, num_classes=2).float().detach()
        # one_hot = stride_weights + one_hot.detach() - stride_weights.detach()
        # print(stride_weights.shape)
        # Select mask using one-hot
        # TODO: is this optimal?
        mask = stride_weights[0] * mask_1 + stride_weights[1] * mask_2
        
        return mask, stride_weights

    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality

        # Calculate query, key, values for all heads in batch
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # Calculate attention scores
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))  # (B, nh, T, T)

        # Get stride weights using softmax
        stride_weights = F.softmax(self.stride_params, dim=-1)
        # print("stride_params", self.stride_params.data)

        # if torch.rand(1) < 0.001:  # Print occasionally (0.1% of the time)
        #     print("stride_params:", self.stride_params.data)
        #     print("stride_weights:", stride_weights.data)
        #     if self.stride_params.grad is not None:
        #         print("stride_params grad:", self.stride_params.grad)
        # print("stride_params", self.stride_params)

        # Clamp span parameters between 0 and max_span using sigmoid
        clamped_spans = torch.sigmoid(self.span_params) * self.max_span
        
        # Generate and apply adaptive attention masks with stride weights
        masks = []
        chosen_mask_indexes = []
        for span, stride_weight in zip(clamped_spans, stride_weights):
            mask, one_hot = self.get_mask(span, T, device=x.device, stride_weights=stride_weight)
            masks.append(mask)
            chosen_mask_indexes.append(one_hot)
        mask = torch.stack(masks)  # (nh, T, T)
        chosen_mask_indexes = torch.stack(chosen_mask_indexes)  # (nh, 2)
        
        # Expand mask for batch dimension
        mask = mask.unsqueeze(0).expand(B, -1, -1, -1)  # (B, nh, T, T)
        
        # Apply mask
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        weighted_exp_att = torch.exp(att) * mask
        att = weighted_exp_att / weighted_exp_att.sum(dim=-1, keepdim=True)
        
        # Apply dropout
        att = self.attn_dropout(att)
        
        # Apply attention to values
        y = att @ v  # (B, nh, T, hs)
        
        # Restore time as batch dimension and concat heads
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        
        # Calculate combined regularization loss
        # weights_stride_loss = 1. / torch.Tensor(self.strides).to(x.device) # (2,)
        # strides_loss = chosen_mask_indexes @ weights_stride_loss.view(-1, 1)  # (nh, 1)
        # loss_per_head = clamped_spans * strides_loss.squeeze(-1) # (nh,)

        span_loss = self.span_reg * clamped_spans.sum() / self.n_head
        
        return y, span_loss

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
        attn_out, span_loss = self.attn(self.ln_1(x))
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, span_loss

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
        for block in self.transformer.h:
            x, span_loss = block(x)
            total_span_loss += span_loss
            
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

        return logits, loss

if __name__ == "__main__":
    # test get mask
    causal_attn = AdaptiveCausalAttention(GPTConfig())
    mask, one_hot = causal_attn.get_mask(0.5, 1024, stride_weights=torch.tensor([0.5, 0.5]))
    print(mask)
    print(one_hot)
