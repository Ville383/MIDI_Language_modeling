import os
import numpy as np
from typing import List
from tqdm import tqdm
from miditok import TSD
from multiprocessing import Pool, cpu_count

# globals inside each worker
_tokenizer = None
_seq_len = None

def _init_worker(tokenizer_path: str, seq_len: int):
    """
    Called once in each worker process.
    Sets up a global tokenizer and seq_len.
    """
    global _tokenizer, _seq_len
    _tokenizer = TSD(params=tokenizer_path)
    _seq_len = seq_len - 2
    if _seq_len <= 0:
        raise ValueError("seq_len must be > 2")

def _worker(midi_path: str):
    """
    Tokenize a single file using the already initialized global tokenizer.
    Returns a list of np.ndarray chunks.
    """
    try:
        seq = _tokenizer(midi_path)[0].ids
        if len(seq) >= _seq_len:
            n_chunks = len(seq) // _seq_len
            return [
                np.array(seq[i * _seq_len : (i + 1) * _seq_len], dtype=np.int16)
                for i in range(n_chunks)
            ]
    except Exception as e:
        print(f"Error processing {midi_path}: {e}")
    return []

def create_dataset(
    file_paths: List[str],
    tokenizer_path: str,
    save_path: str,
    seq_len: int,
    num_workers: int = None,
    save_batch_size: int = 10_000
):
    """
    1) Spins up a Pool where each worker loads the tokenizer once.
    2) Streams results in batches
    3) Merges all parts at the end into a single sequences.npy.
    """
    os.makedirs(save_path, exist_ok=True)

    num_workers = num_workers or cpu_count()

    # prepare for chunked saves
    buffer = []
    part_paths = []
    part_idx = 0
    total_sequences = 0

    def save_chunk():
        nonlocal part_idx, buffer
        if buffer:
            arr = np.stack(buffer)
            p = os.path.join(save_path, f"sequences_part_{part_idx}.npy")
            np.save(p, arr)
            part_paths.append(p)
            #print(f"Saved {len(buffer)} sequences to {p}")
            part_idx += 1
            buffer = []

    with Pool(
        processes=num_workers,
        initializer=_init_worker,
        initargs=(tokenizer_path, seq_len)
    ) as pool:
        for chunks in tqdm(pool.imap_unordered(_worker, file_paths),
                           total=len(file_paths),
                           desc="Tokenizing MIDI files"):
            if chunks:
                buffer.extend(chunks)
                total_sequences += len(chunks)
                if len(buffer) >= save_batch_size:
                    save_chunk()

    # final flush
    save_chunk()

    if total_sequences == 0:
        print("No valid sequences were found.")
        return

    # merge all parts
    print("Merging all part files into one sequences.npy")
    all_arrs = [np.load(p) for p in part_paths]
    final = np.concatenate(all_arrs, axis=0)
    final_path = os.path.join(save_path, "sequences.npy")
    np.save(final_path, final)
    print(f"Merged {total_sequences} sequences into {final_path}")

    # cleanup
    for p in part_paths:
        os.remove(p)