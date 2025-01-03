import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import Dataset

# number of workers in .map() call
num_proc = 8

enc = tiktoken.get_encoding("gpt2")

def load_text_file(filepath):
    """Load text file where each line is a message."""
    with open(filepath, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f if line.strip()]
    return Dataset.from_dict({"text": texts})

if __name__ == '__main__':
    # Load the text file
    input_file = "../../../messages.txt"  # Replace with your input file path
    dataset = load_text_file(input_file)

    # Create train/val split
    split_dataset = dataset.train_test_split(test_size=0.0005, seed=2357, shuffle=True)
    split_dataset['val'] = split_dataset.pop('test')  # rename test split to val

    def process(example):
        """Tokenize text and add EOT token."""
        ids = enc.encode_ordinary(example['text'])
        ids.append(enc.eot_token)
        return {'ids': ids, 'len': len(ids)}

    # Tokenize the dataset
    tokenized = split_dataset.map(
        process,
        remove_columns=['text'],
        desc="tokenizing the splits",
        num_proc=num_proc,
    )

    # Save tokenized data to binary files
    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = os.path.join(os.path.dirname(__file__), f'{split}.bin')
        dtype = np.uint16  # works since enc.max_token_value < 2**16
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))

        # Calculate number of batches based on dataset size
        # Use at most 1024 batches, but fewer for smaller datasets
        total_batches = min(1024, len(dset))
        if total_batches == 0:
            total_batches = 1  # Ensure at least one batch

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()

    # Print some statistics
    print("\nDataset statistics:")
    for split, dset in tokenized.items():
        total_tokens = np.sum(dset['len'])
        print(f"{split}: {len(dset)} messages, {total_tokens:,} tokens")