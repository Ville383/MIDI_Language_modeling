'''
Train a Transformer encoder model to generate MIDI:
    1. pre-train
    (2. SFT)
    (3. RL)
'''
import torch
from miditok import TSD
from config import ModelParams, HyperParams
from utils import MIDIDataset_pre, Trainer, collate_fn_pre
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from itertools import cycle

if __name__ == '__main__':
    tokenizer = TSD(params="models/tokenizer_trained.json")

    TRAIN_DATA_PATH = "pre-training_dataset/train/sequences.npy"
    VAL_DATA_PATH = "pre-training_dataset/val/sequences.npy"

    train_dataset = MIDIDataset_pre(TRAIN_DATA_PATH, tokenizer)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=HyperParams.batch_size, collate_fn=collate_fn_pre, shuffle=True)
    val_dataset = MIDIDataset_pre(VAL_DATA_PATH, tokenizer)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=HyperParams.batch_size, collate_fn=collate_fn_pre, shuffle=True)   

    print(f"number of train token sequences: {len(train_dataset)}")
    print(f"number of val token sequences: {len(val_dataset)}")
    print(f"vocab size: {tokenizer.vocab_size}")

    trainer = Trainer(ModelParams, HyperParams, tokenizer)
    print(sum(p.numel() for p in trainer.mask_predictor.parameters()), "model parameters\n")
    print("Model architecture:")
    print(trainer.mask_predictor)
    if HyperParams.retrain:
        # load model, optimizer, and lr scheduler from a checkpoint (fine_tune=True -> load only model)
        # optionally change global_step manually
        trainer.load("models/checkpoints/pre-training_704000_checkpoint.pth", fine_tune=False)
    
    print(f"batch size: {HyperParams.batch_size}, accumulation: {HyperParams.accumulation_steps}\nretrain: {HyperParams.retrain}\ntraining steps: {HyperParams.num_training_steps}\nlogging every {HyperParams.logging_interval} interval")

    # Setup
    trainer.phase = 'pre-training' # 'pre-training' or 'SFT'
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
                mc_loss = torch.zeros(1, device=trainer.device)
                for i in range(1): # Can increase to get more accurate val loss (but will take longer)
                    mc_loss[i] = trainer.val_step(val_batch)
                losses[k] = mc_loss.mean().item()
            val_loss = losses.mean().item()
            
            trainer.mask_predictor.train()

            writer.add_scalar('Loss/val', val_loss / len(val_dataloader), global_step)       
            writer.add_scalar('Loss/train', train_loss / HyperParams.logging_interval, global_step)
            writer.add_scalar('LR', trainer.lr_scheduler.get_last_lr()[0], global_step)

            if val_loss < best_val:
                trainer.save('models/best_val_checkpoint_pre.pth', val_loss / len(val_dataloader))
                best_val = val_loss

            trainer.save(model_dir + str(global_step) + '_checkpoint.pth', train_loss / HyperParams.logging_interval)
            train_loss = 0
        pbar.update(1)

    pbar.close()
    writer.close()