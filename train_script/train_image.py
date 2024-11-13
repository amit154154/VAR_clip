import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb
from dataset.imagenet_dataset_siglip import get_train_val_datasets  # Updated import
from trainers.var_image_trainer import VAR_Image  # Updated import
from pytorch_lightning.callbacks import ModelCheckpoint

# ------------------- Hyperparameters -------------------

# General settings
do_wandb = False                     # Whether to use Weights & Biases for logging
hugging_face_token = "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"         # Hugging Face token
project_name = 'VAR_siglip_image_alpha'         # WandB project name (updated to reflect SigLIP usage)
seed = 420134                        # Random seed for reproducibility
CKPT_PATH = None                    # Path to a checkpoint to resume training (if any)
CHECKPOINT_DIR = "checkpoints_08"      # Directory to save checkpoints
CHECKPOINT_EVERY_N_TRAIN_STEPS = 2500  # Save checkpoint every N training steps
SAVE_LAST_CHECKPOINT = True         # Whether to save the last checkpoint
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps')  # Device configuration
log_k = 10                          # Interval for logging generated images
do_lora = False
only_class = None

# Training parameters
batch_size = 8                      # Batch size for both training and validation
learning_rate = 1e-3              # Learning rate for the optimizer
num_steps = 20000                   # Number of steps to train
GRADIENT_CLIP_VAL = 1               # Gradient clipping value
PRECISION = '16-mixed'              # Precision mode (e.g., 16-bit mixed precision)
ACCUMULATE_GRAD_BATCHES = 1         # Gradient accumulation steps

# Model paths (update these paths to point to your actual checkpoint files)
var_ckpt_path = '/Users/mac/Downloads/var_d16.pth'            # Path to VAR checkpoint
vae_ckpt_path = '/Users/mac/Downloads/vae_ch160v4096z32.pth'    # Path to VAE checkpoint

# Model settings
clip_model_name = 'openai/clip-vit-base-patch32'               # CLIP model name for tokenizer
siglip_model_name = 'google/siglip-base-patch16-224'               # SigLIP model name for image processor
beta = 1                                                     # Beta parameter for the model
alpha = 0                                                     # Alpha parameter for the model
MODEL_DEPTH = 16                                                 # Depth of the VAR model

# Data parameters
final_reso = 256                                                # Final resolution of images
do_hflip = False                                                # Whether to apply horizontal flip augmentation

# -------------------------------------------------------

# WandB configuration
WANDB_CONFIG = {
    "batch_size": batch_size,
    "learning_rate": learning_rate,
    "clip_model_name": clip_model_name,         # For tokenizer
    "siglip_model_name": siglip_model_name,     # For SigLIP image processor
    "beta": beta,
    "alpha": alpha,
    "seed": seed,
    "do_lora":do_lora,
    "ACCUMULATE_GRAD_BATCHES":ACCUMULATE_GRAD_BATCHES,
    "only_class":only_class
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
    model_name=clip_model_name,             # For CLIP tokenizer
    siglip_model_name=siglip_model_name,     # For SigLIP image processor
    final_reso=final_reso,
    do_hflip=do_hflip,
    batch_size=batch_size,
    hugging_face_token=hugging_face_token,
    only_class=only_class
)

# Initialize the model
model = VAR_Image(
    device=DEVICE,
    MODEL_DEPTH=MODEL_DEPTH,
    var_ckpt=var_ckpt_path,
    vae_ckpt=vae_ckpt_path,
    siglip_model=siglip_model_name,           # Updated parameter
    log_k=log_k,
    do_wandb=do_wandb,
    beta=beta,
    alpha=alpha,
    learning_rate=learning_rate
)

# Apply LoRA to VAR model
if do_lora:
    model.apply_lora_to_var()

# Set up checkpointing
checkpoint_callback = ModelCheckpoint(
    dirpath=CHECKPOINT_DIR,
    filename="model-step-{step}",
    save_top_k=-1,                                   # Save all checkpoints
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

trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=CKPT_PATH)