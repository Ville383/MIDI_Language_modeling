'''
Train a Transformer encoder model to generate MIDI:
    1. pre-train
    (2. SFT)
    (3. RL)
'''
import torch
from pathlib import Path
from miditok import TSD
from config import ModelParams, HyperParams
from utils import MIDIDataset_sft, Trainer, collate_fn_sft
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from itertools import cycle

if __name__ == '__main__':
    tokenizer = TSD(params="models/tokenizer_trained.json")
    pickle_path = Path("sft_dataset/train/data.pkl")
    train_dataset = MIDIDataset_sft(pickle_path)
    pickle_path = Path("sft_dataset/val/data.pkl")
    val_dataset = MIDIDataset_sft(pickle_path)

    val_dataloader = DataLoader(dataset=val_dataset, batch_size=HyperParams.batch_size, collate_fn=collate_fn_sft, shuffle=True)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=HyperParams.batch_size, collate_fn=collate_fn_sft, shuffle=True)

    print(f"number of train token sequences: {len(train_dataset)}")
    print(f"number of val token sequences: {len(val_dataset)}")
    print(f"vocab size: {tokenizer.vocab_size}")

    trainer = Trainer(ModelParams, HyperParams, tokenizer)
    print(sum(p.numel() for p in trainer.mask_predictor.parameters()), "model parameters\n")
    print("Model architecture:")
    print(trainer.mask_predictor)
    trainer.load("models/best_val_checkpoint_pre.pth", fine_tune=True)

    print(f"batch size: {HyperParams.batch_size}, accumulation: {HyperParams.accumulation_steps}\nretrain: {HyperParams.retrain}\ntraining steps: {HyperParams.num_training_steps}\nlogging every {HyperParams.logging_interval} interval")

    # Setup
    trainer.phase = 'SFT' # 'pre-training' or 'SFT'
    model_dir = 'models/checkpoints/' + trainer.phase + '_'
    writer = SummaryWriter()
    train_iterator = cycle(train_dataloader)
    pbar = tqdm(total=HyperParams.num_training_steps, desc="Training", position=0)
    train_loss = 0
    global_step = 0
    best_val = float('inf')
    trainer.mask_predictor.train()
    while global_step < HyperParams.num_training_steps:
        train_batch = next(train_iterator)
        train_loss += trainer.train_step(train_batch)
        global_step += 1

        # Logging
        if global_step % HyperParams.logging_interval == 0:
            trainer.mask_predictor.eval()

            val_loss = 0
            losses = torch.zeros(len(val_dataloader), device=trainer.device)
            for k, val_batch in tqdm(enumerate(val_dataloader), desc="Validating", leave=False):
                mc_loss = torch.zeros(64, device=trainer.device)
                for i in range(64): # Can increase to get more accurate val loss (but will take longer)
                    mc_loss[i] = trainer.val_step(val_batch)
                losses[k] = mc_loss.mean().item()
            val_loss = losses.mean().item()
            
            trainer.mask_predictor.train()

            writer.add_scalar('Loss/val', val_loss / len(val_dataloader), global_step)       
            writer.add_scalar('Loss/train', train_loss / HyperParams.logging_interval, global_step)
            writer.add_scalar('LR', trainer.lr_scheduler.get_last_lr()[0], global_step)

            if val_loss < best_val:
                trainer.save('models/best_val_checkpoint_sft.pth', val_loss / len(val_dataloader))
                best_val = val_loss

            trainer.save(model_dir + str(global_step) + '_checkpoint.pth', train_loss / HyperParams.logging_interval)
            train_loss = 0
        pbar.update(1)

    pbar.close()
    writer.close()