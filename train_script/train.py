import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb
from dataset.imagenet_dataset import  get_train_val_datasets
from trainers.var_text_trainer import VAR_text
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

# ------------------- Hyperparameters -------------------

# General settings
do_wandb = True                # Whether to use Weights & Biases for logging
project_name = 'VAR_clip'  # WandB project name
seed = 4201                     # Random seed for reproducibility
CKPT_PATH = None
CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_EVERY_N_TRAIN_STEPS = 1000
SAVE_LAST_CHECKPOINT = True
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps')
log_k = 25                      # Interval for logging images

# Training parameters
batch_size = 8                # Batch size for both training and validation
learning_rate = 3e-4           # Learning rate for the optimizer
num_steps = 10000              # Number of steps to train
GRADIENT_CLIP_VAL = 1
PRECISION = '16-mixed'  # 16-bit precision
ACCUMULATE_GRAD_BATCHES = 1

# Model paths (update these paths to point to your actual checkpoint files)
var_ckpt_path = '/Users/mac/Downloads/var_d16.pth'  # Path to VAR checkpoint
vae_ckpt_path = '/Users/mac/Downloads/vae_ch160v4096z32.pth'  # Path to VAE checkpoint

# Model settings
clip_model_name = 'openai/clip-vit-base-patch32'  # CLIP model name
beta = 1                      # Beta parameter for the model
alpha = 0                        # Alpha parameter for the model
MODEL_DEPTH = 16                 # Depth of the model

# Data parameters
final_reso = 256                 # Final resolution of images
do_hflip = False                 # Whether to apply horizontal flip augmentation

# -------------------------------------------------------
WANDB_CONFIG = {
    "batch_size": batch_size,
    "learning_rate": learning_rate,
    "clip_model_name":clip_model_name,
    "beta":beta,
    "alpha":alpha,
    "seed":seed

}

# Set random seed for reproducibility
pl.seed_everything(seed)

# Set up the WandB logger if enabled
if do_wandb:
    wandb_logger = WandbLogger(project=project_name, config=WANDB_CONFIG)
else:
    wandb_logger = None

# Get the data loaders
train_dataloader, val_dataloader = get_train_val_datasets(
    model_name=clip_model_name,
    final_reso=final_reso,
    do_hflip=do_hflip,
    batch_size=batch_size
)



# Initialize the model
model = VAR_text(
    device=DEVICE,
    MODEL_DEPTH=MODEL_DEPTH,
    var_ckpt=var_ckpt_path,
    vae_ckpt=vae_ckpt_path,
    clip_model=clip_model_name,
    log_k=log_k,
    do_wandb=do_wandb,
    beta=beta,
    alpha=alpha
)

checkpoint_callback = ModelCheckpoint(
    dirpath=CHECKPOINT_DIR,
    filename="model-step-{step}",
    save_top_k=-1,
    every_n_train_steps=CHECKPOINT_EVERY_N_TRAIN_STEPS,
    save_last=SAVE_LAST_CHECKPOINT
)

# -----------------------
# Set up PyTorch Lightning trainer
# -----------------------

trainer = pl.Trainer(
    logger=wandb_logger,
    log_every_n_steps=5,
    gradient_clip_val=GRADIENT_CLIP_VAL,
    precision=PRECISION,
    callbacks=[checkpoint_callback],
    accumulate_grad_batches=ACCUMULATE_GRAD_BATCHES,
    max_steps=num_steps,
)

# -----------------------
# Start training
# -----------------------

trainer.fit(model, train_dataloader, ckpt_path=CKPT_PATH)