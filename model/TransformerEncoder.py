import torch
import torch.nn as nn
import torch.nn.functional as F
from config import ModelParams
from model.rotary_embedding import RotaryPositionalEmbeddings

class Encoder(nn.Module):
    def __init__(self, args: ModelParams):
        super().__init__()
        self.d_model = args.d_model
        self.dim_ff = args.dim_ff
        self.n_heads = args.n_heads
        self.dropout = args.dropout
        self.bias = args.bias
        
        # Head dimension
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        self.head_dim = self.d_model // self.n_heads
        
        # Model architecture
        self.norm1 = nn.RMSNorm(self.d_model, eps=1e-5)
        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(self.d_model, self.d_model, bias=self.bias)
        self.k_proj = nn.Linear(self.d_model, self.d_model, bias=self.bias)
        self.v_proj = nn.Linear(self.d_model, self.d_model, bias=self.bias)
        self.rope = RotaryPositionalEmbeddings(self.head_dim, max_seq_len=args.max_seq_len)
        self.out_proj = nn.Linear(self.d_model, self.d_model, bias=self.bias)
        self.dropout1 = nn.Dropout(self.dropout)

        self.norm2 = nn.RMSNorm(self.d_model, eps=1e-5)
        # Feedforward network layers
        self.linear1 = nn.Linear(self.d_model, self.dim_ff, bias=self.bias)
        self.activation = nn.GELU()
        self.dropout_ff = nn.Dropout(self.dropout)
        self.linear2 = nn.Linear(self.dim_ff, self.d_model, bias=self.bias)
        self.dropout2 = nn.Dropout(self.dropout)
    
    def forward(self, src):
        batch_size, seq_len, _ = src.shape
        
        # Pre-norm for attention
        norm_src = self.norm1(src)
        
        # Compute Q, K, V
        q = self.q_proj(norm_src)
        k = self.k_proj(norm_src)
        v = self.v_proj(norm_src)
        
        # Reshape for multi-head attention: (batch, seq_len, n_heads, head_dim)
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim)
        
        # Apply RoPE to queries and keys (expects [b, s, n_h, h_d] format)
        q = self.rope(q)
        k = self.rope(k)
        
        # Transpose for scaled_dot_product_attention: (batch, n_heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Apply scaled dot-product attention
        attn_output = F.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            attn_mask=None,
            dropout_p=(self.dropout if self.training else 0.0),
            is_causal=False
        )
        
        # Reshape back: (batch, n_heads, seq_len, head_dim) -> (batch, seq_len, d_model)
        attn_output = attn_output.transpose(1, 2).view(batch_size, seq_len, self.d_model)
        
        # Apply output projection
        attn_output = self.out_proj(attn_output)
        
        # Residual connection and dropout
        src = src + self.dropout1(attn_output)
        
        # Pre-norm for feedforward
        norm_src = self.norm2(src)
        
        # Feedforward network
        ff_output = self.linear2(self.dropout_ff(self.activation(self.linear1(norm_src))))
        
        # Residual connection and dropout
        src = src + self.dropout2(ff_output)
        
        return src


class TokenEmbedding(nn.Module):
  def __init__(self, n_token, d_model):
    super(TokenEmbedding, self).__init__()

    self.n_token = n_token
    self.emb_scale = d_model ** 0.5

    self.emb_lookup = nn.Embedding(n_token, d_model)

  def forward(self, inp_tokens):
    inp_emb = self.emb_lookup(inp_tokens)
    return inp_emb.mul_(self.emb_scale)