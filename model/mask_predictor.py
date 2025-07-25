import torch.nn as nn
from model.TransformerEncoder import Encoder, TokenEmbedding
from miditok import TSD
from config import ModelParams

class MaskPredictor(nn.Module):
    """
    The core LLaDA mask predictor.
    """
    def __init__(self, args: ModelParams, tokenizer: TSD):
        super().__init__()
        self.vocab_size = tokenizer.vocab_size

        # Learned embeddings
        self.token_emb = TokenEmbedding(self.vocab_size, args.d_model)

        # Transformer encoder layers
        self.enc_layers = nn.ModuleList([Encoder(args) for _ in range(args.num_layers)])
        self.lm_head = nn.Linear(args.d_model, self.vocab_size)

        # Weight tying
        self.lm_head.weight = self.token_emb.emb_lookup.weight
        assert self.lm_head.weight.data_ptr() == self.token_emb.emb_lookup.weight.data_ptr()
            
    def forward(self, x_t):
        """
        x_t: [batch_size, seq_len], token IDs (with some positions masked)
        
        Returns:
        logits: a [batch_size, seq_len, vocab_size] tensor of logits.
        """
        # 1) Get embeddings
        hidden = self.token_emb(x_t) # [B x L x D]
        
        # 2) feed through transformer layers
        for layer in self.enc_layers:
          hidden = layer(hidden) # [B x L x D]

        # 3) Compute logits over vocabulary
        logits = self.lm_head(hidden) # [B x L x V]
        return logits