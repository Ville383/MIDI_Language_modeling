import torch
from dataclasses import dataclass

@dataclass
class HyperParams:
    name: str = "transformer_12_layers_512_d_model_8_n_heads_1024_dim_ff"
    batch_size: int = 32
    accumulation_steps: int = 64 // batch_size # 32 used in pre-training
    block_size: int = 512
    lr: float = 1e-4
    retrain: bool = False
    num_training_steps: int = 400000 * accumulation_steps
    num_warmup_steps: int = 16000 # updates every accumulation_steps
    logging_interval: int = 5000 * accumulation_steps


@dataclass
class ModelParams:
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Transformer:
    num_layers: int = 12
    d_model: int = 256
    n_heads: int = 8
    dim_ff: int = 1024
    dropout: float = 0.1
    bias: bool = False
    max_seq_len: int = HyperParams.block_size