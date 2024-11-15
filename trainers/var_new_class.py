import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import SiglipVisionModel  # Removed CLIPTextModel and CLIPTokenizer
from torch.optim import AdamW
from models import VQVAE, build_vae_var
import wandb
from torchvision.transforms import ToPILImage
import torchvision
import numpy as np
import PIL.Image as PImage
from transformers import get_linear_schedule_with_warmup  # Import the scheduler
from peft import LoraConfig, get_peft_model  # Import PEFT components
from torchvision.datasets import ImageNet

class VAR_newclass(pl.LightningModule):
    def __init__(self, device='cpu',
                 MODEL_DEPTH=16,
                 var_ckpt='/Users/mac/Downloads/var_d16.pth',
                 vae_ckpt='/Users/mac/Downloads/vae_ch160v4096z32.pth',
                 log_k=100,
                 do_wandb=False,
                 hugging_face_token=None,
                 learning_rate = 1e-3,
                 start_class_id = 578 # Doll
                 ):
        super().__init__()
        self.device_name = device

        # Initialize VAE and VAR models
        patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
        self.vae, self.var = build_vae_var(
            V=4096, Cvae=32, ch=160, share_quant_resi=4,
            device=device, patch_nums=patch_nums,
            num_classes=1000, depth=MODEL_DEPTH, shared_aln=False,
        )
        self.var.load_state_dict(torch.load(var_ckpt, map_location=device), strict=True)
        self.vae.load_state_dict(torch.load(vae_ckpt, map_location=device), strict=True)
        self.vae.eval()
        self.hugging_face_token = hugging_face_token
        self.learning_rate = learning_rate

        # Freeze all other parameters
        for param in self.vae.parameters():
            param.requires_grad = False

        # Define loss and logging parameters
        self.train_loss = nn.CrossEntropyLoss(label_smoothing=0.1, reduction='mean')
        self.log_k = log_k
        self.do_wandb = do_wandb
        self.start_class_id = start_class_id

    def denormalize_pm1_to_01(self, x):
        """
        Denormalizes the tensor from [-1, 1] to [0, 1].

        Args:
            x (Tensor): Normalized tensor.

        Returns:
            Tensor: Denormalized tensor.
        """
        return x.add(1.0).div(2.0)
    def forward(self, label_B, x_BLCv_wo_first_l):

        logits_BLV = self.var(label_B, x_BLCv_wo_first_l)
        return logits_BLV

    def training_step(self, batch, batch_idx):
        images = batch
        B, V = images.shape[0], self.vae.vocab_size
        class_id  = torch.full((B,), fill_value=self.start_class_id).to(self.device)

        # VAE quantization
        gt_idx_Bl = self.vae.img_to_idxBl(images)
        gt_BL = torch.cat(gt_idx_Bl, dim=1)  # Shape: (B, 680)
        x_BLCv_wo_first_l = self.vae.quantize.idxBl_to_var_input(gt_idx_Bl)

        # Forward pass
        logits_BLV = self(class_id, x_BLCv_wo_first_l)
        loss = self.train_loss(logits_BLV.view(-1, V), gt_BL.view(-1))

        # Generate and log images at specified intervals
        if (self.global_step % self.log_k == 0 or self.global_step == 1) and self.do_wandb:
            self.log_generated_images(images[0])

        # Log the training loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def log_generated_images(self,gt_image):

        seed = 0

        generated_images = self.var.autoregressive_infer_cfg(
            B=1,
            label_B=self.start_class_id,
            top_k=900,
            top_p=0.95,
            more_smooth=False,
            g_seed=seed
        )
        self.logger.experiment.log({
            f"generated_images": wandb.Image(
                generated_images
            ),
            f"gt_image": wandb.Image(
                gt_image
            )
        }, step=self.global_step)

    def configure_optimizers(self):
        """
        Configures the optimizer and scheduler.

        Returns:
            Tuple[List[Optimizer], List[dict]]: Optimizer and scheduler configurations.
        """
        # Collect trainable parameters in self.var (LoRA parameters) and adapter
        optimizer = AdamW(params=self.var.parameters(),
            lr=self.learning_rate,
        )
        total_steps = self.trainer.max_steps  # Access the total number of training steps
        warmup_steps = int(0.08 * total_steps)  # Set warmup steps (e.g., 1% of total steps)

        # Create the scheduler
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        # Return both the optimizer and the scheduler
        scheduler_config = {
            'scheduler': scheduler,
            'interval': 'step',  # Update the scheduler every step
            'frequency': 1
        }
        return [optimizer], [scheduler_config]