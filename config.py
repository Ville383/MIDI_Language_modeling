from dataclasses import dataclass

@dataclass
class HyperParams:
    DATA_PATH: str = "you/dataset/path"
    name: str = "transformer_10_layers_512_d_model_8_n_heads_2048_dim_ff"
    batch_size: int = 8
    accumulation_steps: int = 16 // batch_size # 16 virtual batch size
    lr: float = 4e-4
    retrain: bool = False
    num_training_steps: int = 200000 * accumulation_steps  # Accumulation
    num_warmup_steps: int = 20000
    logging_interval: int = 8000


@dataclass
class ModelParams:
    # Transformer:
    num_layers: int = 10
    d_model: int = 512
    n_heads: int = 8
    dim_ff: int = 2048
    dropout: float = 0.1
    max_seq_len: int = 2048
    bias: bool = False