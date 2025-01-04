import time
import os
import numpy as np

# First, let's check the size of our dataset to set appropriate parameters
data_dir = os.path.join('data', 'allie')
train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

# Calculate block size based on dataset size
# Use 90% of the smallest file's length to ensure we can always sample sequences
min_data_length = min(len(train_data), len(val_data))
block_size = min(1024, max(1, min_data_length // 2))  # Don't exceed 1024, ensure at least 1

print(f"Dataset sizes: train={len(train_data)}, val={len(val_data)}")
print(f"Using block_size={block_size}")

out_dir = 'out-allie'
eval_interval = 5
eval_iters = 40
wandb_log = True  # feel free to turn on
wandb_project = 'allie'
wandb_run_name = 'ft-' + str(time.time())

dataset = 'allie'
init_from = 'gpt2-xl'  # this is the largest GPT-2 model

# only save checkpoints if the validation loss improves
always_save_checkpoint = False

# Adjust batch size and gradient accumulation based on dataset size
batch_size = 1
gradient_accumulation_steps = min(32, max(1, min_data_length // (block_size * 2)))
max_iters = 20

# finetune at constant LR
learning_rate = 3e-5
decay_lr = False

# Add additional safety parameters
# Dropout can help with small datasets
dropout = 0.1