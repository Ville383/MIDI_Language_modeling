import math
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from config import ModelParams
from miditok import REMI

class MaskPredictor(nn.Module):
    """
    The core LLaDA mask predictor.
    """
    def __init__(self, args: ModelParams, tokenizer: REMI):
        super().__init__()
        self.vocab_size = tokenizer.vocab_size
        self.d_model = args.d_model
        self.dim_ff = args.dim_ff
        self.n_heads = args.n_heads
        self.n_layers = args.num_layers
        self.dropout = args.dropout
        self.bias = args.bias
        self.mask_token_id = tokenizer.vocab["MASK_None"]
        self.max_seq_len = 2048 # ~1300 max length in training set

        # Learned embeddings
        self.token_emb = TokenEmbedding(self.vocab_size, self.d_model, self.d_model)
        self.pos_emb = PositionalEncoding(self.d_model)
        self.emb_dropout = nn.Dropout(self.dropout)

        # Transformer encoder layers
        encoder_layer = TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            dim_feedforward=self.dim_ff,
            dropout=self.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
            bias=self.bias
        )
        self.transformer = TransformerEncoder(encoder_layer, self.n_layers)

        self.lm_head = nn.Linear(self.d_model, self.vocab_size)

    def forward(self, x_t):
        """
        x_t: [batch_size, seq_len], token IDs (with some positions masked)
        
        Returns:
        logits: a [batch_size, seq_len, vocab_size] tensor of logits.
        """
        batch_size, seq_len = x_t.shape

        # 1) Get embeddings
        token_embeddings = self.token_emb(x_t) # [B x L x D]
        pos_embeddings = self.pos_emb(seq_len, batch_size) # [B x L x D]

        # 2) Add embeddings and dropout
        hidden = self.emb_dropout(token_embeddings + pos_embeddings)
        
        # 3) feed through transformer layers
        hidden = self.transformer(hidden) # [B x L x D]

        # 4) Compute logits over vocabulary
        logits = self.lm_head(hidden) # [B x L x V]
        return logits


class PositionalEncoding(nn.Module):
    def __init__(self, d_embed, max_pos=20480):
        super(PositionalEncoding, self).__init__()
        self.d_embed = d_embed
        self.max_pos = max_pos

        pe = torch.zeros(max_pos, d_embed)
        position = torch.arange(0, max_pos, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_embed, 2).float() * (-math.log(10000.0) / d_embed))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe) # shape: (1, max_pos, d_embed)

    def forward(self, seq_len, bsz):
        pos_encoding = self.pe[:, :seq_len, :]

        return pos_encoding.expand(bsz, seq_len, self.d_embed) # shape: (batch_size, seq_len, d_embed)


class TokenEmbedding(nn.Module):
  def __init__(self, n_token, d_embed, d_proj):
    super(TokenEmbedding, self).__init__()

    self.n_token = n_token
    self.d_embed = d_embed
    self.d_proj = d_proj
    self.emb_scale = d_proj ** 0.5

    self.emb_lookup = nn.Embedding(n_token, d_embed)
    if d_proj != d_embed:
      self.emb_proj = nn.Linear(d_embed, d_proj, bias=False)
    else:
      self.emb_proj = None

  def forward(self, inp_tokens):
    inp_emb = self.emb_lookup(inp_tokens)
    
    if self.emb_proj is not None:
      inp_emb = self.emb_proj(inp_emb)

    return inp_emb.mul_(self.emb_scale)