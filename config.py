from dataclasses import dataclass

@dataclass
class HyperParams:
    DATA_PATH = "you/dataset/path"
    name = "transformer_12_layers_256_d_model_8_n_heads"
    epochs: int = 1
    batch_size: int = 2
    accumulation_steps: int = 16 // batch_size # 16 'virtual' batch size
    lr = 4e-4
    retrain: bool = False
    num_warmup_steps: int = 2000
    num_training_steps: int = 1000000 # optimizer steps = num_training_steps // accumulation_steps
    logging_interval: int = num_training_steps // 400


@dataclass
class ModelParams:
    # Transformer:
    num_layers: int = 12
    d_model: int = 256
    n_heads: int = 8
    dim_ff: int = 1024
    dropout: float = 0.1
    max_seq_len: int = 2048