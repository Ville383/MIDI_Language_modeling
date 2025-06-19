from pathlib import Path
import pickle
import random
from typing import List
from tqdm import tqdm
from miditok import REMI
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

def create_bar_dataset(
    midi_paths: List[Path],
    tokenizer: REMI,
    output_dir: Path,
):
    """
    Creates a dataset of tokenized MIDI segments of melodic phrases and its accompaniment. Uses the POP909 dataset (midi tracks 0 and 2).
    """
    def get_bar_indices(track):
        indices = {}
        bar_idx = -1
        for i, token in enumerate(track):
            if token == tokenizer['Bar_None']:
                bar_idx += 1
                indices[bar_idx] = i
        return indices
    
    # Ensure the output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    file_counter = 0

    for midi_path in tqdm(midi_paths, desc="Processing MIDI files"):
        try:
            # Tokenize the MIDI file. Returns a list of token sequences (one per track).
            midi = tokenizer(midi_path)
        except Exception as e:
            print(f"Skipping {midi_path.name} due to a processing error: {e}")
            continue

        melody_track = midi[0]
        acc_track = midi[2]

        acc_bar_indices = get_bar_indices(acc_track)
        melody_bar_indices = get_bar_indices(melody_track)

        # Find non-empty melody bars for phrase detection. There must also be accompaniment bars present.
        melody_bars = {}
        acc_bars_set = set(acc_bar_indices)
        for bar, token_idx in melody_bar_indices.items():
            if bar in acc_bars_set and token_idx + 1 < len(melody_track) and melody_track[token_idx + 1] != tokenizer['Bar_None']:
                melody_bars[bar] = token_idx

        sorted_melody_bars = sorted(melody_bars.keys())

        if not sorted_melody_bars:
            continue  # Skip this file, no usable bars to phrase

        phrases = []
        current_phrase = [sorted_melody_bars[0]]

        for i in range(1, len(sorted_melody_bars)):
            current_bar = sorted_melody_bars[i]
            last_bar_in_phrase = current_phrase[-1]

            # Check if the current bar is 1 or 2 steps from the last
            if current_bar == last_bar_in_phrase + 1 or current_bar == last_bar_in_phrase + 2:
                current_phrase.append(current_bar)
            else:
                # Sequence broken. Save phrase if it's long enough.
                if len(current_phrase) >= 3:
                    phrases.append(current_phrase)
                current_phrase = [current_bar]

        # Check the last phrase
        if len(current_phrase) >= 3:
            phrases.append(current_phrase)

        for i, phrase_bars in enumerate(phrases):
            input_ids = []
            target_ids = []

            # Iterate through each bar number in the potentially gapped phrase
            for bar_num in phrase_bars:
                next_bar_num = bar_num + 1
                
                # Get token indices for this bar using the full bar maps
                melody_start_idx = melody_bar_indices[bar_num]
                acc_start_idx = acc_bar_indices[bar_num]
                
                # End index is the start of the next bar, or track end if it's the last one
                melody_end_idx = melody_bar_indices.get(next_bar_num, len(melody_track))
                acc_end_idx = acc_bar_indices.get(next_bar_num, len(acc_track))

                # Append the tokens for this single bar
                input_ids.extend(melody_track[melody_start_idx:melody_end_idx])
                target_ids.extend(acc_track[acc_start_idx:acc_end_idx])

            # Prepare data for pickling and transform it to correct structure for diffusion.
            data_to_save = {
                "input_ids": input_ids,
                "input_bar_start_indices": [i for i, token in enumerate(input_ids) if token == tokenizer['Bar_None']],
                "target_ids": target_ids,
                "target_bar_start_indices": [i for i, token in enumerate(target_ids) if token == tokenizer['Bar_None']],
                "num_bars": sum(1 for bar in input_ids if bar == tokenizer['Bar_None']),
            }

            # Save to a pickle file
            output_path = output_dir / f"sample_{file_counter}.pkl"
            with open(output_path, "wb") as f:
                pickle.dump(data_to_save, f)
            
            file_counter += 1


class MIDIDataset(Dataset):
    def __init__(self, pickle_paths, tokenizer, pitch_aug=True):
        self.pickle_paths = pickle_paths
        self.pitch_aug = pitch_aug
        self.tokenizer = tokenizer
        self.pitch_ids = [val for key, val in tokenizer.vocab.items() if key.startswith("Pitch_")] # [3, 87]
        self.min_pitch = min(self.pitch_ids)
        self.max_pitch = max(self.pitch_ids)
        self.num_bars_options = [4, 5, 6, 7, 8]
        self.bar_probs = [0.4, 0.3, 0.05, 0.05, 0.2]

    def __len__(self):
        return len(self.pickle_paths)

    def __getitem__(self, idx):
        with open(self.pickle_paths[idx], 'rb') as f:
            data = pickle.load(f)
        
        input_ids = data["input_ids"]
        target_ids = data["target_ids"]
        input_bar_start_indices = data["input_bar_start_indices"]
        target_bar_start_indices = data["target_bar_start_indices"]

        num_bars = random.choices(self.num_bars_options, weights=self.bar_probs)[0]

        if len(input_bar_start_indices) > num_bars:
            input_start_index = input_bar_start_indices[0]
            input_end_index = input_bar_start_indices[num_bars]
            selected_input_ids = input_ids[input_start_index:input_end_index]

            target_start_index = target_bar_start_indices[0]
            target_end_index = target_bar_start_indices[num_bars]
            selected_target_ids = target_ids[target_start_index:target_end_index]
        else:
            # Not enough bars: return whole input
            selected_input_ids = input_ids
            selected_target_ids = target_ids
        
        ids = selected_input_ids + selected_target_ids

        if self.pitch_aug:
            shift = random.randint(-6, 6)
            for i in range(len(ids)):
                if ids[i] in self.pitch_ids:
                    shifted = ids[i] + shift
                    if self.min_pitch <= shifted <= self.max_pitch:
                        ids[i] = shifted
                  
        return torch.tensor(ids, dtype=torch.long), len(selected_input_ids), len(ids)

class DataCollator:
    def __init__(self, pad_token_id=0):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        input_ids, prompt_lengths, lengths = zip(*batch)
        padded_tokens = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        # EXPECTED FORMAT:
        # "input_ids": [prompt + answer], "input_length": prompt length, "length": [prompt + answer] length   
        return {
            'input_ids': padded_tokens,
            'input_length': torch.tensor(prompt_lengths, dtype=torch.long),
            'length': torch.tensor(lengths, dtype=torch.long),
        }