import time
import os
import numpy as np

# First, let's check the size of our dataset to set appropriate parameters
data_dir = os.path.join('data', 'allie')
train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

# Calculate block size based on dataset size
min_data_length = min(len(train_data), len(val_data))
block_size = min(1024, max(1, min_data_length // 2))

# Calculate tokens per iteration
batch_size = 1
gradient_accumulation_steps = min(32, max(1, min_data_length // (block_size * 2)))
tokens_per_iter = batch_size * gradient_accumulation_steps * block_size

# Calculate iterations needed for one epoch (one full pass through data)
iters_per_epoch = len(train_data) / tokens_per_iter
desired_epochs = 3  # Adjust this number to control how many times to go through the data
max_iters = int(iters_per_epoch * desired_epochs)

print(f"Dataset sizes: train={len(train_data)}, val={len(val_data)} tokens")
print(f"Using block_size={block_size}")
print(f"Processing {tokens_per_iter} tokens per iteration")
print(f"One epoch = {iters_per_epoch:.1f} iterations")
print(f"Training for {desired_epochs} epochs = {max_iters} iterations")

out_dir = 'out-allie'
eval_interval = 5
eval_iters = 40
wandb_log = True
wandb_project = 'allie'
wandb_run_name = 'ft-' + str(time.time())

dataset = 'allie'
init_from = 'gpt2-xl'

always_save_checkpoint = False
learning_rate = 3e-5
decay_lr = False
dropout = 0.1