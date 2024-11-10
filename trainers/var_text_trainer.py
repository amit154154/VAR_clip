import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import CLIPTextModel, CLIPTokenizer
from torch.optim import AdamW
from models import VQVAE, build_vae_var
import wandb
from torchvision.transforms import ToPILImage
import torch, torchvision
import numpy as np
import PIL.Image as PImage, PIL.ImageDraw as PImageDraw
from transformers import get_linear_schedule_with_warmup  # Import the scheduler
from peft import LoraConfig, get_peft_model  # Import PEFT components
from torchvision.datasets import ImageNet

imagenet_class_names = {i: f"Class_{i}" for i in range(1000)}  # Placeholder mapping

class SimpleAdapter(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=1024, out_dim=1024):
        super(SimpleAdapter, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.norm0 = nn.modules.normalization.LayerNorm(input_dim)
        self.activation1 = nn.GELU(approximate='tanh')
        self.layer2 = nn.Linear(hidden_dim, out_dim)
        self.norm2 = nn.modules.normalization.LayerNorm(out_dim)
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


class VAR_text(pl.LightningModule):
    def __init__(self, device='cpu',
                 MODEL_DEPTH=16,
                 var_ckpt='/Users/mac/Downloads/var_d16.pth',
                 vae_ckpt='/Users/mac/Downloads/vae_ch160v4096z32.pth',
                 clip_model = 'openai/clip-vit-base-patch32',
                 log_k=100,
                 do_wandb=False,
                 beta=0.01,
                 alpha = 1
                 ):
        super().__init__()
        self.device_name = device
        patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
        self.vae, self.var = build_vae_var(
            V=4096, Cvae=32, ch=160, share_quant_resi=4,
            device=device, patch_nums=patch_nums,
            num_classes=1000, depth=MODEL_DEPTH, shared_aln=False,
        )
        self.var.load_state_dict(torch.load(var_ckpt, map_location=device), strict=True)
        self.vae.load_state_dict(torch.load(vae_ckpt, map_location=device), strict=True)
        self.vae.eval()

        # Initialize the CLIP text encoder and set to eval mode
        self.clip_text_encoder = CLIPTextModel.from_pretrained(clip_model).to(device)
        self.clip_text_encoder.eval()

        # Tokenizer if needed for text inputs
        self.tokenizer = CLIPTokenizer.from_pretrained(clip_model)

        # Initialize improved adapter with custom initialization and normalization
        self.adapter = SimpleAdapter().to(device)

        # Freeze all other parameters
        for param in self.vae.parameters():
            param.requires_grad = False
        for param in self.clip_text_encoder.parameters():
            param.requires_grad = False
        for param in self.var.parameters():
             param.requires_grad = False

        self.train_loss = nn.CrossEntropyLoss(label_smoothing=0.1, reduction='mean')
        self.log_k = log_k
        self.do_wandb = do_wandb

        self.beta = beta
        self.alpha = alpha

    def apply_lora_to_var(self):
        # Apply LoRA to self.var after loading the checkpoint
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

    def forward(self, label_B, x_BLCv_wo_first_l, description):
        # Move description to the correct device
        description = {k: v.to(self.device_name) for k, v in description.items()}
        # Normalize the CLIP encoding
        cond_delta = self.clip_text_encoder(**description).pooler_output
        cond_delta = F.normalize(cond_delta, p=2, dim=-1)  # L2 normalization

        # Pass through the adapter
        cond_delta = self.adapter(cond_delta)

        return self.var(label_B, x_BLCv_wo_first_l, cond_delta, beta=self.beta,alpha=self.alpha), cond_delta

    def denormalize_pm1_to_01(self, x):
        return x.add(1.0).div(2.0)

    def training_step(self, batch, batch_idx):
        images, description, class_id = batch['images'], batch['captions'], batch['labels']
        B, V = class_id.shape[0], self.vae.vocab_size

        gt_idx_Bl = self.vae.img_to_idxBl(images)
        gt_BL = torch.cat(gt_idx_Bl, dim=1)  # (b,680)
        x_BLCv_wo_first_l = self.vae.quantize.idxBl_to_var_input(gt_idx_Bl)

        logits_BLV, cond_delta = self(class_id, x_BLCv_wo_first_l, description)
        loss = self.train_loss(logits_BLV.view(-1, V), gt_BL.view(-1))

        # Generate and log images only at specified intervals
        if self.global_step % self.log_k == 0 and self.do_wandb:
            self.log_generated_images(class_id, cond_delta, description)

        # Log the training loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def log_generated_images(self, class_id, cond_delta, description):
        """
        Logs generated images with different alpha and beta settings to WandB.

        Args:
            class_id: The class ID of the image.
            cond_delta: Conditioning delta from the CLIP text encoder.
            description: Text description input for conditioning.
        """
        # Settings for different alpha and beta combinations
        settings = [
            {'alpha': 1, 'beta': 0, 'label': 'Alpha=1 and Beta=0'},
            {'alpha': self.alpha, 'beta': self.beta, 'label': 'Alpha = alpha and Beta = beta'},
            {'alpha': 1, 'beta': self.beta, 'label': 'Alpha=1 and Beta = beta'},
            {'alpha': 1, 'beta': self.beta * 2, 'label': 'Alpha=1 and Beta = 2*beta'},
        ]

        images_to_log = []
        seed = random.randint(0,100000)
        for setting in settings:
            if setting['beta'] == 0:
                generated_images = self.var.autoregressive_infer_cfg(
                    B=1,
                    label_B=class_id[:1],
                    cond_delta=cond_delta[:1],
                    beta=setting['beta'],
                    alpha=setting['alpha'],
                    top_k=900,
                    top_p=0.95,
                    more_smooth=False,
                    g_seed = seed
                )
            else:
                generated_images = self.var.autoregressive_infer_cfg(
                    B=1,
                    label_B=class_id[:1],
                    cond_delta=None,
                    beta=setting['beta'],
                    alpha=setting['alpha'],
                    top_k=900,
                    top_p=0.95,
                    more_smooth=False,
                    g_seed=seed

                )

            # Convert to PIL image
            image = ToPILImage()(generated_images[0])
            images_to_log.append({'image': image, 'label': setting['label']})

        # Detokenize description and get class name for the first item in the batch
        detokenized_description = self.tokenizer.decode(description['input_ids'][0], skip_special_tokens=True)
        class_id_first = class_id[0].item()
        class_name = imagenet_class_names.get(class_id_first, "Unknown Class")

        # Log the images to WandB with captions including class name and description
        for img_info in images_to_log:
            caption = f"{class_name}: {detokenized_description} ({img_info['label']})"
            self.logger.experiment.log({
                f"generated_images/{img_info['label']}": wandb.Image(
                    img_info['image'],
                    caption=caption
                )
            }, step=self.global_step)

    def generate_example(self, beta, text, class_id, B=1, seed=None,cfg=0,more_smooth=False):
        """
        Generate example images given beta, text, and class_id.

        Args:
            beta (float): The beta parameter controlling the sampling.
            text (str or list of str): The input text description(s).
            class_id (int): The class ID for conditioning.
            B (int): The batch size (number of images to generate).
            seed (int, optional): Random seed for reproducibility.

        Returns:
            PIL.Image.Image: The generated image(s) as a PIL image.
        """
        with torch.no_grad():
            # Set the random seed if provided
            if seed is not None:
                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)
                np.random.seed(seed)

            # Prepare the text descriptions
            if isinstance(text, str):
                text = [text] * B  # Duplicate text for batch
            description = self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt"
            )
            # Move description to device
            description = {k: v.to(self.device) for k, v in description.items()}

            # Get cond_delta via clip_text_encoder and adapter
            cond_delta = self.clip_text_encoder(**description).pooler_output
            cond_delta = F.normalize(cond_delta, p=2, dim=-1)  # L2 normalization
            cond_delta = self.adapter(cond_delta)

            # Prepare beta
            beta_tensor = torch.tensor(beta, dtype=torch.float32).to(self.device_name)
            # Prepare label_B
            label_B = torch.tensor([class_id] * B, dtype=torch.long).to(self.device)

            # Generate images
            recon_B3HW = self.var.autoregressive_infer_cfg(
                B=B,
                label_B=label_B,
                cfg=cfg,  # Assuming cfg is 0.0
                top_k=900,
                top_p=0.95,
                g_seed=seed,
                more_smooth=more_smooth,
                beta=beta_tensor,
                cond_delta=cond_delta,
            )

            # Convert the generated images to a grid
            chw = torchvision.utils.make_grid(recon_B3HW, nrow=8, padding=0, pad_value=1.0)
            chw = chw.permute(1, 2, 0).mul_(255).cpu().numpy()
            image = PImage.fromarray(chw.astype(np.uint8))

            return image

    def configure_optimizers(self):
        # Collect trainable parameters in self.var (LoRA parameters)
        var_trainable_params = [p for p in self.var.parameters() if p.requires_grad]
        optimizer = AdamW(
            [{'params': self.adapter.parameters()},
             {'params': var_trainable_params}],
            lr=3e-4,
        )
        total_steps = self.trainer.max_steps  # Access the total number of training steps
        warmup_steps = int(0.01 * total_steps)  # Set warmup steps (e.g., 1% of total steps)

        # Create the scheduler
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        # Return both the optimizer and the scheduler
        scheduler = {
            'scheduler': scheduler,
            'interval': 'step',  # Update the scheduler every step
            'frequency': 1
        }
        return [optimizer], [scheduler]