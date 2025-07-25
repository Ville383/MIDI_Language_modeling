import numpy as np
import torch
from torch.utils.data import Dataset
from torch.amp import GradScaler, autocast
import torch.nn.functional as F
from transformers import get_cosine_schedule_with_warmup
from model.mask_predictor import MaskPredictor
from config import HyperParams, ModelParams
from miditok import TSD
import pickle
from pathlib import Path

# Trainer class for mask predictor
class Trainer:
    def __init__(self, m_args: ModelParams, h_args: HyperParams, tokenizer: TSD):
        self.m_args = m_args
        self.h_args = h_args
        self.device_str = m_args.device_str
        self.device = torch.device(self.device_str)
        self.mask_predictor = MaskPredictor(m_args, tokenizer)
        self.mask_predictor.to(self.device)
        self.optimizer = torch.optim.AdamW(self.mask_predictor.parameters(), lr=h_args.lr, betas=(0.9, 0.999), weight_decay=0.01)
        self.lr_scheduler = get_cosine_schedule_with_warmup(
                                self.optimizer,
                                num_warmup_steps=h_args.num_warmup_steps,
                                num_training_steps=h_args.num_training_steps // h_args.accumulation_steps
                            )
        self.scaler = GradScaler(device=self.device_str)
        self.tokenizer = tokenizer
        self.name = h_args.name
        self.phase = 'pre-training'
        self.step = 0 # for gradient accumulation
    
    def forward_process(self, input_ids, eps=1e-3):
        b, l = input_ids.shape
        t = torch.rand(b, device=input_ids.device)

        p_mask = (1 - eps) * t + eps
        p_mask = p_mask[:, None].repeat(1, l)

        masked_indices = torch.rand((b, l), device=input_ids.device) < p_mask
        noisy_batch = torch.where(masked_indices, self.tokenizer["MASK_None"], input_ids)
        return noisy_batch, masked_indices, p_mask

    # https://github.com/ML-GSAI/SMDM/tree/main
    def train_step(self, train_data):
        if self.step == 0:
            self.optimizer.zero_grad()
        
        with autocast(device_type=self.device_str):
            if self.phase == 'pre-training':
                input_ids = train_data['input_ids'].to(self.device)

                if torch.rand(1) < 0.01:
                    length = torch.randint(1, self.h_args.block_size + 1, (1,))
                    input_ids = input_ids[:, :length]

                noisy_batch, masked_indices, p_mask = self.forward_process(input_ids)
                logits = self.mask_predictor(noisy_batch)

                token_loss = F.cross_entropy(logits[masked_indices], input_ids[masked_indices], reduction='none') / p_mask[masked_indices]
                loss = token_loss.sum() / (input_ids.shape[0] * input_ids.shape[1])

            if self.phase == 'SFT':
                input_ids = train_data['input_ids'].to(self.device) # [prompt + answer + padding], length=1024
                prompt_lengths = train_data['prompt_lengths'].to(self.device)  # prompt length
                length = train_data['lengths'].to(self.device) # [prompt + answer] length
                max_length = length.max().item()
                input_ids = input_ids[:, :max_length]

                noisy_batch, _, p_mask = self.forward_process(input_ids)

                # Do not add noise to the prompt
                token_positions = torch.arange(noisy_batch.size(1), device=noisy_batch.device).expand(noisy_batch.size(0), noisy_batch.size(1))
                prompt_mask = (token_positions < prompt_lengths.unsqueeze(1))
                noisy_batch[prompt_mask] = input_ids[prompt_mask].clone()

                # Calculate the answer length (including the padded tokens)
                prompt_mask = prompt_mask.to(torch.int64)
                answer_lengths = torch.sum((1 - prompt_mask), dim=-1, keepdim=True)
                answer_lengths = answer_lengths.repeat(1, noisy_batch.shape[1])

                # Identify the tokens that were masked in the answer part
                masked_indices = (noisy_batch == self.tokenizer['MASK_None'])
                
                logits = self.mask_predictor(noisy_batch)
                
                # Calculate loss only for the masked positions
                token_loss = F.cross_entropy(logits[masked_indices], input_ids[masked_indices], reduction='none') / p_mask[masked_indices]
                
                # Normalize by the actual number of tokens being predicted.
                loss = torch.sum(token_loss / answer_lengths[masked_indices]) / input_ids.shape[0]

        loss = loss / self.h_args.accumulation_steps
        self.scaler.scale(loss).backward()

        
        self.step += 1
        if self.step % self.h_args.accumulation_steps == 0:
            prev_scale = self.scaler.get_scale()

            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.mask_predictor.parameters(), max_norm=3.0)

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

            # Check if the optimizer step was skipped
            if self.scaler.get_scale() >= prev_scale:
                self.lr_scheduler.step()
        
        return loss.item()

    # https://github.com/ML-GSAI/SMDM/tree/main
    def val_step(self, val_data):
        with torch.no_grad(), autocast(device_type=self.device_str):
            if self.phase == 'pre-training':
                input_ids = val_data['input_ids'].to(self.device)
                noisy_batch, masked_indices, p_mask = self.forward_process(input_ids)
                logits = self.mask_predictor(noisy_batch)

                token_loss = F.cross_entropy(logits[masked_indices], input_ids[masked_indices], reduction='none') / p_mask[masked_indices]
                loss = token_loss.sum() / (input_ids.shape[0] * input_ids.shape[1])

            if self.phase == 'SFT':
                input_ids = val_data['input_ids'].to(self.device) # [prompt + answer + padding], length=1024
                prompt_lengths = val_data['prompt_lengths'].to(self.device)  # prompt length
                length = val_data['lengths'].to(self.device) # [prompt + answer] length
                max_length = length.max().item()
                input_ids = input_ids[:, :max_length]

                noisy_batch, _, p_mask = self.forward_process(input_ids)

                token_positions = torch.arange(noisy_batch.size(1), device=noisy_batch.device).expand(noisy_batch.size(0), noisy_batch.size(1))
                prompt_mask = (token_positions < prompt_lengths.unsqueeze(1))
                noisy_batch[prompt_mask] = input_ids[prompt_mask].clone()

                prompt_mask = prompt_mask.to(torch.int64)
                answer_lengths = torch.sum((1 - prompt_mask), dim=-1, keepdim=True)
                answer_lengths = answer_lengths.repeat(1, noisy_batch.shape[1])

                masked_indices = (noisy_batch == self.tokenizer['MASK_None'])

                logits = self.mask_predictor(noisy_batch)
                
                token_loss = F.cross_entropy(logits[masked_indices], input_ids[masked_indices], reduction='none') / p_mask[masked_indices]
                loss = torch.sum(token_loss / answer_lengths[masked_indices]) / input_ids.shape[0]
        
        return loss.item()
    
    def save(self, path_name, loss):
        torch.save({
            'model_state_dict': self.mask_predictor.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.lr_scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'loss': loss,
        }, path_name)

    def load(self, path_name, fine_tune=False):
        checkpoint = torch.load(path_name, map_location=self.device)
        self.mask_predictor.load_state_dict(checkpoint['model_state_dict'])
        self.mask_predictor.to(self.device)
        self.mask_predictor.lm_head.weight = self.mask_predictor.token_emb.emb_lookup.weight

        # Create new optimizer for fine-tuning
        if fine_tune:
            self.optimizer = torch.optim.AdamW(self.mask_predictor.parameters(), lr=self.h_args.lr, betas=(0.9, 0.999), weight_decay=0.01)
            self.lr_scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.h_args.num_warmup_steps,
                num_training_steps=self.h_args.num_training_steps // self.h_args.accumulation_steps
            )
            self.scaler = GradScaler(device=self.device_str)
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])


class MIDIDataset_pre(Dataset):
    def __init__(self, token_path: str, tokenizer: TSD, mmap: bool = True):
        load_kwargs = dict(mmap_mode="r") if mmap else {}
        self.tokens = np.load(token_path, **load_kwargs)
        self.num_samples = self.tokens.shape[0]

        self.bos = np.array([tokenizer["BOS_None"]], dtype=self.tokens.dtype)
        self.eos = np.array([tokenizer["PAD_None"]], dtype=self.tokens.dtype)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int): 
        tokens_with_special = np.concatenate([self.bos, self.tokens[idx], self.eos])
        return torch.from_numpy(tokens_with_special).long()

def collate_fn_pre(batch):
    return {'input_ids': torch.stack(batch)}


class MIDIDataset_sft(Dataset):
    def __init__(self, pickle_path: Path):
        with open(pickle_path, "rb") as f:
            data = pickle.load(f)
        self.input_ids = data["input_ids"]
        self.prompt_lengths = data["prompt_lengths"]
        self.lengths = data["lengths"]
        self.length = data["dataset_length"]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.input_ids[idx], self.prompt_lengths[idx], self.lengths[idx]

def collate_fn_sft(batch):
    input_ids, prompt_lengths, lengths = zip(*batch)
    return {
        'input_ids': torch.stack(input_ids),
        'prompt_lengths': torch.stack(prompt_lengths),
        'lengths': torch.stack(lengths),
    }