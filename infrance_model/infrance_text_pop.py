from trainers.var_image_trainer import SimpleAdapter
import torch
from models import VQVAE, build_vae_var
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, SiglipTextModel
from peft import LoraConfig, get_peft_model
import random
from torchvision.transforms import ToPILImage
import numpy as np
from moviepy.editor import ImageSequenceClip
import random
import gradio as gr
import tempfile
import os


class InrenceTextVAR(nn.Module):
    def __init__(self, pl_checkpoint=None, start_class_id=578, hugging_face_token=None, siglip_model='google/siglip-base-patch16-224', device="cpu", MODEL_DEPTH=16):
        super(InrenceTextVAR, self).__init__()
        self.device = device
        self.class_id = start_class_id
        # Define layers
        patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
        self.vae, self.var = build_vae_var(
            V=4096, Cvae=32, ch=160, share_quant_resi=4,
            device=device, patch_nums=patch_nums,
            num_classes=1000, depth=MODEL_DEPTH, shared_aln=False,
        )
        self.text_processor = AutoTokenizer.from_pretrained(siglip_model, token=hugging_face_token)
        self.siglip_text_encoder = SiglipTextModel.from_pretrained(siglip_model, token=hugging_face_token).to(device)
        self.adapter = SimpleAdapter(
            input_dim=self.siglip_text_encoder.config.hidden_size,
            out_dim=self.var.C  # Ensure dimensional consistency
        ).to(device)
        self.apply_lora_to_var()
        if pl_checkpoint is not None:
            state_dict = torch.load(pl_checkpoint, map_location="cpu")['state_dict']
            var_state_dict = {k[len('var.'):]: v for k, v in state_dict.items() if k.startswith('var.')}
            vae_state_dict = {k[len('vae.'):]: v for k, v in state_dict.items() if k.startswith('vae.')}
            adapter_state_dict = {k[len('adapter.'):]: v for k, v in state_dict.items() if k.startswith('adapter.')}
            self.var.load_state_dict(var_state_dict)
            self.vae.load_state_dict(vae_state_dict)
            self.adapter.load_state_dict(adapter_state_dict)
        del self.vae.encoder

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

    @torch.no_grad()
    def generate_image(self, text, beta=1, seed=None, more_smooth=False, top_k=0, top_p=0.9):
        if seed is None:
            seed = random.randint(0, 2**32 - 1)
        inputs = self.text_processor([text], padding="max_length", return_tensors="pt").to(self.device)
        outputs = self.siglip_text_encoder(**inputs)
        pooled_output = outputs.pooler_output  # pooled (EOS token) states
        pooled_output = F.normalize(pooled_output, p=2, dim=-1)  # Normalize delta condition
        cond_delta = F.normalize(pooled_output, p=2, dim=-1).to(self.device)  # Use correct device
        cond_delta = self.adapter(cond_delta)
        cond_delta = F.normalize(cond_delta, p=2, dim=-1)  # Normalize delta condition
        generated_images = self.var.autoregressive_infer_cfg(
            B=1,
            label_B=self.class_id,
            delta_condition=cond_delta[:1],
            beta=beta,
            alpha=1,
            top_k=top_k,
            top_p=top_p,
            more_smooth=more_smooth,
            g_seed=seed
        )
        image = ToPILImage()(generated_images[0].cpu())
        return image

    @torch.no_grad()
    def generate_video(self, text, start_beta, target_beta, fps, length, top_k=0, top_p=0.9, seed=None,
                       more_smooth=False,
                       output_filename='output_video.mp4'):

        if seed is None:
            seed = random.randint(0, 2 ** 32 - 1)

        num_frames = int(fps * length)
        images = []

        # Define an easing function for smoother interpolation
        def ease_in_out(t):
            return t * t * (3 - 2 * t)

        # Generate t values between 0 and 1
        t_values = np.linspace(0, 1, num_frames)
        # Apply the easing function
        eased_t_values = ease_in_out(t_values)
        # Interpolate beta values using the eased t values
        beta_values = start_beta + (target_beta - start_beta) * eased_t_values

        for beta in beta_values:
            image = self.generate_image(text, beta=beta, seed=seed, more_smooth=more_smooth, top_k=top_k, top_p=top_p)
            images.append(np.array(image))

        # Create a video from images
        clip = ImageSequenceClip(images, fps=fps)
        clip.write_videofile(output_filename, codec='libx264')

if __name__ == '__main__':
    import torch
    from torch.quantization import quantize_dynamic
    import torch.nn as nn

    # Initialize the model
    pl_checkpoint = '/Users/mac/Downloads/model-step-step=35000.ckpt'  # Replace with your actual checkpoint path
    device = 'mps'
    model = InrenceTextVAR(pl_checkpoint=pl_checkpoint, device=device)
    model.to(device)


    def generate_image_gradio(text, beta=1.0, seed=None, more_smooth=False, top_k=0, top_p=0.9):
        print(f"Generating image for text: {text}\n"
              f"beta: {beta}\n"
              f"seed: {seed}\n"
              f"more_smooth: {more_smooth}\n"
              f"top_k: {top_k}\n"
              f"top_p: {top_p}\n")
        image = model.generate_image(text, beta=beta, seed=seed, more_smooth=more_smooth, top_k=int(top_k), top_p=top_p)
        return image

    def generate_video_gradio(text, start_beta=1.0, target_beta=1.0, fps=10, length=5.0, top_k=0, top_p=0.9, seed=None, more_smooth=False, progress=gr.Progress()):
        print(f"Generating video for text: {text}\n"
              f"start_beta: {start_beta}\n"
              f"target_beta: {target_beta}\n"
              f"seed: {seed}\n"
              f"more_smooth: {more_smooth}\n"
              f"top_k: {top_k}\n"
              f"top_p: {top_p}"
              f"fps: {fps}\n"
              f"length: {length}\n")
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmpfile:
            output_filename = tmpfile.name
        num_frames = int(fps * length)
        beta_values = np.linspace(start_beta, target_beta, num_frames)
        images = []

        for i, beta in enumerate(beta_values):
            image = model.generate_image(text, beta=beta, seed=seed, more_smooth=more_smooth, top_k=top_k, top_p=top_p)
            images.append(np.array(image))
            # Update progress
            progress((i + 1) / num_frames)
            # Yield the frame image to update the GUI
            yield image, gr.update()

        # After generating all frames, create the video
        clip = ImageSequenceClip(images, fps=fps)
        clip.write_videofile(output_filename, codec='libx264')

        # Yield the final video output
        yield gr.update(), output_filename

    with gr.Blocks() as demo:
        gr.Markdown("# Text to Image/Video Generator")
        with gr.Tab("Generate Image"):
            text_input = gr.Textbox(label="Input Text")
            beta_input = gr.Slider(label="Beta", minimum=0.0, maximum=2.5, step=0.05, value=1.0)
            seed_input = gr.Number(label="Seed", value=None)
            more_smooth_input = gr.Checkbox(label="More Smooth", value=False)
            top_k_input = gr.Number(label="Top K", value=0)
            top_p_input = gr.Slider(label="Top P", minimum=0.0, maximum=1.0, step=0.01, value=0.9)
            generate_button = gr.Button("Generate Image")
            image_output = gr.Image(label="Generated Image")
            generate_button.click(
                generate_image_gradio,
                inputs=[text_input, beta_input, seed_input, more_smooth_input, top_k_input, top_p_input],
                outputs=image_output
            )

        with gr.Tab("Generate Video"):
            text_input_video = gr.Textbox(label="Input Text")
            start_beta_input = gr.Slider(label="Start Beta", minimum=0.0, maximum=2.5, step=0.05, value=0)
            target_beta_input = gr.Slider(label="Target Beta",minimum=0.0, maximum=2.5, step=0.05, value=1.0)
            fps_input = gr.Number(label="FPS", value=10)
            length_input = gr.Number(label="Length (seconds)", value=5.0)
            seed_input_video = gr.Number(label="Seed", value=None)
            more_smooth_input_video = gr.Checkbox(label="More Smooth", value=False)
            top_k_input_video = gr.Number(label="Top K", value=0)
            top_p_input_video = gr.Slider(label="Top P", minimum=0.0, maximum=1.0, step=0.01, value=0.9)
            generate_video_button = gr.Button("Generate Video")
            frame_output = gr.Image(label="Current Frame")
            video_output = gr.Video(label="Generated Video")

            generate_video_button.click(
                generate_video_gradio,
                inputs=[text_input_video, start_beta_input, target_beta_input, fps_input, length_input, top_k_input_video, top_p_input_video, seed_input_video, more_smooth_input_video],
                outputs=[frame_output, video_output],
                queue=True  # Enable queuing to allow for progress updates
            )

    demo.launch()