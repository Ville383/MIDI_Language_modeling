import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from transformers import get_linear_schedule_with_warmup
import random

from config import ModelParams, HyperParams
from model.mask_predictor import MaskPredictor



# DATASET FUNCTIONS/CLASSES
# Compute the average rhythmic intensity and polyphony score for each bar. Apply pitch augmentation if used
def compute_cond(event, tokenizer, pitch_aug):
    #n_max_bars = event.ids.count(tokenizer.vocab["Bar_None"])
    #poly_record = np.zeros((n_max_bars * 16,)) # Each bar has 16 position markers (notes can be held to next bar(s) -> flatten to vector)
    #onset_record = np.zeros((n_max_bars, 16))
    #cur_bar, cur_pos = -1, -1

    if pitch_aug:
        shift = random.randint(-6, 6)

    for i, ev in enumerate(event.tokens):
        '''
        if ev == "Bar_None":
            cur_bar += 1
        elif ev.startswith("Position_"):
            cur_pos = int(ev.split("_")[1])
        # polyphony
        elif ev.startswith("Duration_"):
            beat_str, pos_str, res_str = ev.split("_")[1].split(".")
            beat = int(beat_str)
            pos = int(pos_str)
            res = int(res_str)
            time_div = tokenizer.time_division # tokenizer.time_division = 4
            duration = beat * time_div + int((pos / res) * time_div)
            st = cur_bar * 16 + cur_pos
            poly_record[st:st + duration] += 1
        # intensity
        '''
        if ev.startswith("Pitch_"):
            #onset_record[cur_bar, cur_pos] += 1
            # do pitch augmentation
            if pitch_aug:
                pitch = int(ev.split("_")[1])
                shifted = pitch + shift
                if tokenizer.config.pitch_range[0] <= shifted < tokenizer.config.pitch_range[1]:
                    event.ids[i] = tokenizer.vocab[f"Pitch_{shifted}"]

    return event.ids#, poly_record.reshape(-1, 16).mean(axis=1), onset_record.mean(axis=1)


class MIDIDataset(Dataset):
    def __init__(self, midi_paths, tokenizer, pitch_aug=True):
        self.midi_paths = midi_paths
        self.pitch_aug = pitch_aug
        self.tokenizer = tokenizer
        #self.rhym_intensity_bounds = [0.2, 0.25, 0.32, 0.38, 0.44, 0.5, 0.63] # from https://arxiv.org/pdf/2105.04090
        #self.polyphonicity_bounds = [2.63, 3.06, 3.50, 4.00, 4.63, 5.44, 6.44]

    def __len__(self):
        return len(self.midi_paths)

    def __getitem__(self, idx):
        midi = self.tokenizer(self.midi_paths[idx])[0]
        
        ids = compute_cond(midi, self.tokenizer, self.pitch_aug)
        #ids, polyph_raw, rhythm_raw# = compute_cond(midi, self.tokenizer, self.pitch_aug)
        #ids, _, _ = compute_cond(midi, self.tokenizer, self.pitch_aug)
        #polyph_cls = np.searchsorted(self.polyphonicity_bounds, polyph_raw)
        #rfreq_cls = np.searchsorted(self.rhym_intensity_bounds, rhythm_raw)

        return torch.tensor(ids, dtype=torch.long)#, torch.from_numpy(polyph_cls), torch.tensor(rfreq_cls)


class DataCollator:
    def __init__(self, pad_token_id=0):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        #input_ids, polyph_cls, rfreq_cls = zip(*batch)
        input_ids = batch

        #poly_lengths = torch.tensor([len(seq) for seq in polyph_cls])
        #rhythm_lengths = torch.tensor([len(seq) for seq in rfreq_cls])
        padded_tokens = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        #polyph_cls = pad_sequence(polyph_cls, batch_first=True, padding_value=self.pad_token_id)
        #rfreq_cls = pad_sequence(rfreq_cls, batch_first=True, padding_value=self.pad_token_id)

        return {
            'input_ids': padded_tokens,
        }


# Trainer class for mask predictor
class Trainer:
    def __init__(self, m_args: ModelParams, h_args: HyperParams, tokenizer):
        self.m_args = m_args
        self.h_args = h_args
        self.device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(self.device_str)
        self.mask_predictor = MaskPredictor(m_args, tokenizer)
        self.mask_predictor.to(self.device)
        self.optimizer = torch.optim.AdamW(self.mask_predictor.parameters(), lr = h_args.lr)
        self.lr_scheduler = get_linear_schedule_with_warmup(
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
                noisy_batch, masked_indices, p_mask = self.forward_process(input_ids)

                logits = self.mask_predictor(noisy_batch)

                token_loss = F.cross_entropy(logits[masked_indices], input_ids[masked_indices], reduction='none') / p_mask[masked_indices]
                loss = token_loss.sum() / (input_ids.shape[0] * input_ids.shape[1])

            if self.phase == 'SFT':
                input_ids = train_data['input_ids'].to(self.device) # [prompt + answer + padding]
                prompt_lengths = train_data['input_length'].to(self.device) # prompt length
                length = train_data['length'] # [prompt + answer] length
                if length <= self.mask_predictor.max_seq_len:
                    return None

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
            #torch.nn.utils.clip_grad_norm_(self.mask_predictor.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
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
                input_ids = val_data['input_ids'].to(self.device) # [prompt + answer + padding]
                prompt_lengths = val_data['input_length'].to(self.device) # prompt length
                length = val_data['length'] # [prompt + answer] length
                if length <= self.mask_predictor.max_seq_len:
                    return None

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

        return loss.item()
    
    def save(self, path_name, loss):
        torch.save({
            'model_state_dict': self.mask_predictor.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.lr_scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'loss': loss,
        }, f'{path_name}_checkpoint.pth')

    def load(self, path_name, fine_tune=False):
        checkpoint = torch.load(path_name, map_location=self.device)
        self.mask_predictor.load_state_dict(checkpoint['model_state_dict'])
        self.mask_predictor.to(self.device)

        # Create new optimizer for fine-tuning (SFT)
        if fine_tune:
            self.optimizer = torch.optim.AdamW(self.mask_predictor.parameters(), lr=self.h_args.lr)
            self.lr_scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.h_args.num_warmup_steps,
                num_training_steps=self.h_args.num_training_steps // self.h_args.accumulation_steps
            )
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        return checkpoint['loss']