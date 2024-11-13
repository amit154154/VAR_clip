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

# Placeholder mapping for ImageNet classes
imagenet_class_names = {i: f"Class_{i}" for i in range(1000)}

class SimpleAdapter(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=1024, out_dim=1024):
        super(SimpleAdapter, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.norm0 = nn.LayerNorm(input_dim)
        self.activation1 = nn.GELU()
        self.layer2 = nn.Linear(hidden_dim, out_dim)
        self.norm2 = nn.LayerNorm(out_dim)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.norm0(x)
        x = self.layer1(x)
        x = self.activation1(x)
        x = self.layer2(x)
        x = self.norm2(x)
        return x

class VAR_Image(pl.LightningModule):
    def __init__(self, device='cpu',
                 MODEL_DEPTH=16,
                 var_ckpt='/Users/mac/Downloads/var_d16.pth',
                 vae_ckpt='/Users/mac/Downloads/vae_ch160v4096z32.pth',
                 siglip_model='facebook/siglip-vision-base',  # Specify the SigLIP model path or name
                 log_k=100,
                 do_wandb=False,
                 beta=0.01,
                 alpha=1,
                 hugging_face_token=None,
                learning_rate = 1e-3
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

        # Initialize the SigLIP vision encoder and set to eval mode
        self.siglip_vision_encoder = SiglipVisionModel.from_pretrained(siglip_model,token=hugging_face_token).to(device)
        self.siglip_vision_encoder.eval()

        # Initialize the adapter with input dimension based on SigLIP's hidden size
        self.adapter = SimpleAdapter(input_dim=self.siglip_vision_encoder.config.hidden_size).to(device)

        # Freeze all other parameters
        for param in self.vae.parameters():
            param.requires_grad = False
        for param in self.siglip_vision_encoder.parameters():
            param.requires_grad = False
        for param in self.var.parameters():
            param.requires_grad = False

        # Define loss and logging parameters
        self.train_loss = nn.CrossEntropyLoss(label_smoothing=0.1, reduction='mean')
        self.log_k = log_k
        self.do_wandb = do_wandb
        self.beta = beta
        self.alpha = alpha

    def apply_lora_to_var(self):
        """
        Applies LoRA (Low-Rank Adaptation) to the VAR model.
        """
        def find_linear_module_names(model):
            linear_module_names = []
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear):
                    linear_module_names.append(name)
            return linear_module_names

        linear_module_names = find_linear_module_names(self.var)

        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=linear_module_names,
            lora_dropout=0.05,
            bias="none",
        )

        self.var = get_peft_model(self.var, lora_config)
        self.var.train()

    def forward(self, label_B, x_BLCv_wo_first_l, images_siglip):
        """
        Forward pass of the model.

        Args:
            label_B (Tensor): Class labels.
            x_BLCv_wo_first_l (Tensor): VAE quantized inputs.
            images_siglip (Tensor): Condition images processed by SigLIP's processor.

        Returns:
            Tuple[Tensor, Tensor]: Output logits and conditioning delta.
        """
        # Encode images_siglip using SigLIPVisionModel
        with torch.no_grad():
            vision_outputs = self.siglip_vision_encoder(pixel_values=images_siglip)
            # Depending on SigLIP's architecture, choose the appropriate output
            # Here, assuming 'pooler_output' is available; adjust if different
            if hasattr(vision_outputs, 'pooler_output') and vision_outputs.pooler_output is not None:
                cond_delta = vision_outputs.pooler_output
            else:
                # Fallback to last hidden state mean pooling
                cond_delta = vision_outputs.last_hidden_state.mean(dim=1)
            cond_delta = F.normalize(cond_delta, p=2, dim=-1)  # L2 normalization

        # Pass through the adapter
        cond_delta = self.adapter(cond_delta)

        # Forward through the VAR model
        logits_BLV = self.var(label_B, x_BLCv_wo_first_l, cond_delta, beta=self.beta, alpha=self.alpha)

        return logits_BLV, cond_delta

    def denormalize_pm1_to_01(self, x):
        """
        Denormalizes the tensor from [-1, 1] to [0, 1].

        Args:
            x (Tensor): Normalized tensor.

        Returns:
            Tensor: Denormalized tensor.
        """
        return x.add(1.0).div(2.0)

    def training_step(self, batch, batch_idx):
        """
        Training step for each batch.

        Args:
            batch (dict): Batch containing 'images', 'images_siglip', 'labels', and 'description'.
            batch_idx (int): Index of the batch.

        Returns:
            Tensor: Computed loss.
        """
        images, images_siglip, class_id, description = batch['images'], batch['images_siglip'], batch['labels'], batch['description']
        B, V = class_id.shape[0], self.vae.vocab_size

        # VAE quantization
        gt_idx_Bl = self.vae.img_to_idxBl(images)
        gt_BL = torch.cat(gt_idx_Bl, dim=1)  # Shape: (B, 680)
        x_BLCv_wo_first_l = self.vae.quantize.idxBl_to_var_input(gt_idx_Bl)

        # Forward pass
        logits_BLV, cond_delta = self(label_B=class_id, x_BLCv_wo_first_l=x_BLCv_wo_first_l, images_siglip=images_siglip)
        loss = self.train_loss(logits_BLV.view(-1, V), gt_BL.view(-1))

        # Generate and log images at specified intervals
        if self.global_step % self.log_k == 0 and self.do_wandb:
            self.log_generated_images(class_id, cond_delta, images_siglip)

        # Log the training loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def log_generated_images(self, class_id, cond_delta, images_siglip):
        """
        Logs generated images with different alpha and beta settings to WandB.

        Args:
            class_id (Tensor): The class IDs of the images.
            cond_delta (Tensor): Conditioning delta from the SigLIP vision encoder.
            images_siglip (Tensor): Condition images processed by SigLIP's processor.
        """
        # Define different settings for alpha and beta
        settings = [
            {'alpha': self.alpha, 'beta': self.beta, 'label': f'Beta=beta-{self.beta}'},
            {'alpha': self.alpha, 'beta': 1.5*self.beta, 'label': f'Beta=1.5*beta-{1.5*self.beta}'},
            {'alpha': 1, 'beta': 0, 'label': 'alpha = 1 Beta=0'}
        ]

        images_to_log = []
        seed = random.randint(0, 100000)
        for setting in settings:
            # Adjust cond_delta based on beta
            generated_images = self.var.autoregressive_infer_cfg(
                B=1,
                label_B=class_id[:1],
                cond_delta=cond_delta[:1],
                beta=setting['beta'],
                alpha=setting['alpha'],
                top_k=900,
                top_p=0.95,
                more_smooth=False,
                g_seed=seed
            )


            # Convert to PIL image
            image = ToPILImage()(generated_images[0].cpu())
            images_to_log.append({'image': image, 'label': setting['label']})
        images_to_log.append({'image': images_siglip[0], 'label': "original image"})
        # Retrieve class name for logging
        class_id_first = class_id[0].item()
        class_name = imagenet_class_names.get(class_id_first, "Unknown Class")

        # Log each generated image to WandB with appropriate captions
        for img_info in images_to_log:
            caption = f"{class_name}: Generated with {img_info['label']}"
            self.logger.experiment.log({
                f"generated_images/{img_info['label']}": wandb.Image(
                    img_info['image'],
                    caption=caption
                )
            }, step=self.global_step)

    def generate_example(self, beta, condition_image, class_id, B=1, seed=None, cfg=0, more_smooth=False):
        """
        Generates example images given beta, condition_image, and class_id.

        Args:
            beta (float): The beta parameter controlling the sampling.
            condition_image (PIL.Image.Image or Tensor): The conditioning image.
            class_id (int): The class ID for conditioning.
            B (int, optional): The batch size (number of images to generate).
            seed (int, optional): Random seed for reproducibility.
            cfg (float, optional): Configuration parameter (default is 0).
            more_smooth (bool, optional): Smoothing parameter (default is False).

        Returns:
            PIL.Image.Image: The generated image as a PIL image.
        """
        with torch.no_grad():
            # Set the random seed if provided
            if seed is not None:
                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)
                np.random.seed(seed)

            # Prepare the condition image
            if isinstance(condition_image, PImage.Image):
                condition_image = [condition_image] * B  # Duplicate for batch
            elif isinstance(condition_image, torch.Tensor):
                condition_image = [condition_image] * B

            # Assuming condition_image is already processed by SiglipImageProcessor
            condition_images_processed = torch.stack(condition_image).to(self.device)

            # Encode condition images using SigLIPVisionModel
            vision_outputs = self.siglip_vision_encoder(pixel_values=condition_images_processed)
            if hasattr(vision_outputs, 'pooler_output') and vision_outputs.pooler_output is not None:
                cond_delta = vision_outputs.pooler_output
            else:
                cond_delta = vision_outputs.last_hidden_state.mean(dim=1)
            cond_delta = F.normalize(cond_delta, p=2, dim=-1)  # L2 normalization

            # Pass through the adapter
            cond_delta = self.adapter(cond_delta)

            # Prepare beta tensor
            beta_tensor = torch.tensor(beta, dtype=torch.float32).to(self.device_name)

            # Prepare label tensor
            label_B = torch.tensor([class_id] * B, dtype=torch.long).to(self.device)

            # Generate images using VAR
            recon_B3HW = self.var.autoregressive_infer_cfg(
                B=B,
                label_B=label_B,
                cond_delta=cond_delta,
                beta=beta_tensor,
                alpha=self.alpha,
                cfg=cfg,  # Assuming cfg is 0.0
                top_k=900,
                top_p=0.95,
                g_seed=seed,
                more_smooth=more_smooth,
            )

            # Convert the generated images to a grid and then to a PIL image
            chw = torchvision.utils.make_grid(recon_B3HW, nrow=8, padding=0, pad_value=1.0)
            chw = chw.permute(1, 2, 0).mul_(255).cpu().numpy()
            image = PImage.fromarray(chw.astype(np.uint8))

            return image

    def configure_optimizers(self):
        """
        Configures the optimizer and scheduler.

        Returns:
            Tuple[List[Optimizer], List[dict]]: Optimizer and scheduler configurations.
        """
        # Collect trainable parameters in self.var (LoRA parameters) and adapter
        var_trainable_params = [p for p in self.var.parameters() if p.requires_grad]
        optimizer = AdamW(
            [{'params': self.adapter.parameters()},
             {'params': var_trainable_params}],
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