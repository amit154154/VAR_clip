import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import AutoTokenizer, SiglipTextModel
from torch.optim import AdamW
from models import VQVAE, build_vae_var
import wandb
from torchvision.transforms import ToPILImage
import torchvision
import numpy as np
import PIL.Image as PImage
from transformers import get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model

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
                nn.init.xavier_uniform_(m.weight, gain=0.001)
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
                 pl_image_checkpoint = None,
                 siglip_model='google/siglip-base-patch16-224',
                 log_k=100,
                 do_wandb=False,
                 beta=1,
                 alpha=1,
                 hugging_face_token=None,
                 learning_rate=1e-5,  # Reduced learning rate
                 start_class_id=578,
                 do_lora = False
                 ):
        super().__init__()

        self.start_class_id = start_class_id
        self.hugging_face_token = hugging_face_token
        self.learning_rate = learning_rate
        self.train_loss = nn.CrossEntropyLoss(reduction='mean')  # Removed label smoothing
        self.log_k = log_k
        self.do_wandb = do_wandb
        self.beta_target = beta
        self.alpha = alpha
        self.do_lora = do_lora


        # Initialize VAE and VAR models
        patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
        self.vae, self.var = build_vae_var(
            V=4096, Cvae=32, ch=160, share_quant_resi=4,
            device=device, patch_nums=patch_nums,
            num_classes=1000, depth=MODEL_DEPTH, shared_aln=False,
        )
        # Initialize the SigLIP vision encoder and set to eval mode
        self.text_processor = AutoTokenizer.from_pretrained(siglip_model)
        self.siglip_text_encoder = SiglipTextModel.from_pretrained(siglip_model).to(device)
        self.adapter = SimpleAdapter(
            input_dim=self.siglip_text_encoder.config.hidden_size,
            out_dim=self.var.C  # Ensure dimensional consistency
        ).to(device)

        if pl_image_checkpoint is not None:
            state_dict= torch.load(pl_image_checkpoint, map_location="cpu")['state_dict']
            var_state_dict = {k[len('var.'):]: v for k, v in state_dict.items() if k.startswith('var.')}
            vae_state_dict = {k[len('vae.'):]: v for k, v in state_dict.items() if k.startswith('vae.')}
            adapter_state_dict = {k[len('adapter.'):]: v for k, v in state_dict.items() if k.startswith('adapter.')}

            self.apply_lora_to_var()
            self.var.load_state_dict(var_state_dict)
            self.vae.load_state_dict(vae_state_dict)
            self.adapter.load_state_dict(adapter_state_dict)

        # Freeze all other parameters
        for param in self.vae.parameters():
            param.requires_grad = False
        for param in self.siglip_text_encoder.parameters():
            param.requires_grad = False
        if not do_lora:
            for param in self.var.parameters():
                param.requires_grad = False

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

    def forward(self, label_B, x_BLCv_wo_first_l, text_tokenize, beta):
        """
        Forward pass of the model.

        Args:
            label_B: Tensor of shape (B,), class labels.
            x_BLCv_wo_first_l: Tensor, VAE quantized inputs.
            images_siglip: Tensor, images processed by SigLIP's processor.
            beta: Scalar, current beta value.

        Returns:
            logits_BLV: Output logits.
            cond_delta: Conditioning delta.
        """
        # Encode images_siglip using SigLIPVisionModel
        with torch.no_grad():
            text_outputs = self.siglip_text_encoder(**text_tokenize)
            cond_delta = F.normalize(text_outputs.pooler_output, p=2, dim=-1)

        # Pass through the adapter
        cond_delta = self.adapter(cond_delta)
        cond_delta = F.normalize(cond_delta, p=2, dim=-1)  # Normalize delta condition
        # Forward through the VAR model
        logits_BLV = self.var(
            label_B, x_BLCv_wo_first_l, cond_delta, beta=beta, alpha=self.alpha
        )

        return logits_BLV, cond_delta

    def training_step(self, batch, batch_idx):
        """
        Training step for each batch.

        Args:
            batch: Batch containing 'images' and 'images_siglip'.
            batch_idx: Index of the batch.

        Returns:
            loss: Computed loss.
        """
        images, text_tokenize = batch
        B, V = images.shape[0], self.vae.vocab_size
        class_id = torch.full((B,), fill_value=self.start_class_id).to(self.device)

        # VAE quantization
        gt_idx_Bl = self.vae.img_to_idxBl(images)
        gt_BL = torch.cat(gt_idx_Bl, dim=1)  # Shape: (B, 680)
        x_BLCv_wo_first_l = self.vae.quantize.idxBl_to_var_input(gt_idx_Bl)

        beta = self.beta_target  # Fixed beta value

        # Forward pass
        logits_BLV, cond_delta = self(
            label_B=class_id,
            x_BLCv_wo_first_l=x_BLCv_wo_first_l,
            text_tokenize=text_tokenize,
            beta=beta
        )
        loss = self.train_loss(logits_BLV.view(-1, V), gt_BL.view(-1))

        # Log the training loss and beta
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("beta", beta, on_step=True, on_epoch=False, prog_bar=True, logger=True)

        # Add logging every log_k steps
        if self.global_step % self.log_k == 0 and self.do_wandb:
            self.log_generated_images(class_id, cond_delta, images, beta,text_tokenize)

        return loss

    def log_generated_images(self, class_id, cond_delta, images, current_beta,text_tokenize):
        """
        Logs generated images with different alpha and beta settings to WandB.

        Args:
            class_id: The class IDs of the images.
            cond_delta: Conditioning delta from the SigLIP vision encoder.
            images_siglip: Condition images processed by SigLIP's processor.
            current_beta: The current beta value used in training.
        """
        # Define different settings for alpha and beta
        input_ids = text_tokenize['input_ids'][0][0].tolist()

        # Detokenize and skip special tokens
        detokenized_text = self.text_processor.decode(input_ids, skip_special_tokens=True)

        settings = [
            {'alpha': self.alpha, 'beta': 0, 'label': f'beta=0'},
            {'alpha': self.alpha, 'beta': current_beta, 'label': f'beta={current_beta}'},
            {'alpha': self.alpha, 'beta': 1.2 * current_beta, 'label': f'beta={1.2 * current_beta}'}
        ]

        images_to_log = []
        seed = random.randint(0, 100000)
        for setting in settings:
            # Generate images using VAR
            generated_images = self.var.autoregressive_infer_cfg(
                B=1,
                label_B=class_id[:1],
                delta_condition=cond_delta[:1],
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

        # Also log the ground truth image (first in the batch)
        # If images are normalized, denormalize them
        gt_image = images[0].cpu()
        if gt_image.min() < 0:
            gt_image = gt_image * 0.5 + 0.5  # Assuming images are in [-1, 1]
        gt_image = ToPILImage()(gt_image)
        images_to_log.append({'image': gt_image, 'label': 'Ground Truth Image'})

        # Log each generated image to WandB with appropriate captions
        for img_info in images_to_log:
            caption = detokenized_text
            self.logger.experiment.log({
                f"generated_images/{img_info['label']}": wandb.Image(
                    img_info['image'],
                    caption=caption
                )
            }, step=self.global_step)

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
        warmup_steps = int(0.1 * total_steps)  # Set warmup steps (e.g., 8% of total steps)

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