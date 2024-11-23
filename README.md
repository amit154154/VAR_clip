# Text-to-Image Generation with Visual Autoregressive Model (VAR)

This repository is an implementation of a text-to-image model inspired by the Visual Autoregressive (VAR) paradigm described in the [Visual Autoregressive Modeling Paper](https://arxiv.org/abs/2404.02905). This model leverages a combination of BLIP-2 for image captioning, a frozen VAE for image representation, and an adapter trained to conditionally adjust class embeddings with textual descriptions.

## Overview

### Architecture
1. **BLIP-2 for Captioning:** Generates captions and class descriptions for images, creating an initial dataset.
2. **VAR and VAE with Adapter:**
   - The VAE and VAR layers are frozen.
   - An adapter layer is trained to map the text-encoded descriptions (via a CLIP text encoder) to a conditional adjustment (`cond_delta`) for the VAR, influencing its output based on image captions.

<div style="display: flex; align-items: center; flex-wrap: wrap; gap: 10px;">
  <h3 style="margin: 0;">PopYOU Experiment</h3>
  <a href="https://huggingface.co/AmitIsraeli/VARpop" style="text-decoration: none;">
    <img src="https://img.shields.io/badge/Hugging%20Face-Model-orange" alt="Hugging Face Model">
  </a>
  <a href="https://huggingface.co/spaces/AmitIsraeli/PopYou" style="text-decoration: none;">
    <img src="https://img.shields.io/badge/Hugging%20Face-Space-purple" alt="Hugging Face Space">
  </a>
  <a href="https://api.wandb.ai/links/amit154154/yhev15mj" style="text-decoration: none;">
    <img src="https://img.shields.io/badge/Weights%20%26%20Biases-Report-blue" alt="Weights & Biases Report">
  </a>
</div>

The **PopYOU Experiment** focuses on generating high-quality Funko Pop! images through the following steps:

1. **Dataset Generation:**
   - **Comprehensive Dataset Creation:** Generated a dataset of approximately **100,000 Funko Pop! images** using detailed prompts.
   - **High-Quality Data:** Utilized **SDXL Turbo** to ensure high-quality data creation, enhancing the dataset's reliability and diversity.

2. **Model Fine-Tuning:**
   - **VAR Model Adaptation:** Fine-tuned the **Visual AutoRegressive (VAR) model**, which was pretrained on ImageNet, specifically adapting it for **Funko Pop! generation**.
   - **Custom Embedding Injection:** Injected a **custom embedding** representing the "doll" class to specialize the model for generating Funko Pop! styled images.

3. **Adapter Training:**
   - **Frozen SigLIP Image Encoder:** Utilized the **SigLIP image encoder** in a frozen state to maintain its integrity and performance.
   - **Lightweight LoRA Module:** Trained a lightweight **LoRA module** to effectively map image embeddings to text representations within a large language model.

4. **Text-to-Image Generation:**
   - **Encoder Replacement:** Enabled text-to-image generation by **replacing the SigLIP image encoder with its text encoder**.
   - **Efficiency and Quality:** Retained frozen components such as the **VAE** and **generator** to ensure both efficiency and high-quality image generation.

## Future Improvements

- **Training on More Datasets:** Instead of fine-tuning on a new class, remove the class embedding and train on just text.

## Acknowledgments

The architecture and model are inspired by the Visual Autoregressive (VAR) model from the paper [Visual Autoregressive Modeling](https://arxiv.org/abs/2404.02905). I recently came across [VAR-CLIP: Text-to-Image Generator with Visual Auto-Regressive Modeling](https://arxiv.org/abs/2408.01181), which integrates VAR with CLIP for text-to-image generation. For more detailed insights on VAR-CLIP’s contributions, please refer to the paper.

## License

This project is licensed under the “Do Whatever You Want” License. You are free to use, modify, distribute, remix, and even teach this code to a koala if you’d like. No restrictions, no strings attached—just pure coding freedom for all! 🐨