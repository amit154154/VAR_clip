# Text-to-Image Generation with Visual Autoregressive Model (VAR)

This repository is an implementation of a text-to-image model inspired by the Visual Autoregressive (VAR) paradigm described in [Visual Autoregressive Modeling Paper](https://arxiv.org/abs/2404.02905) [oai_citation:3,2404.02905v2.pdf]. This model leverages a combination of BLIP-2 for image captioning, a frozen VAE for image representation, and an adapter trained to conditionally adjust class embeddings with textual descriptions.

## Overview

### Architecture
1. **BLIP-2 for Captioning:** Generates captions and class descriptions for images, creating an initial dataset.
2. **VAR and VAE with Adapter:**
   - The VAE and VAR layers are frozen.
   - An adapter layer is trained to map the text-encoded descriptions (via a CLIP text encoder) to a conditional adjustment (`cond_delta`) for the VAR, influencing its output based on image captions.
   - Conditional input to the VAR is a weighted combination:
     ```
     condition = alpha * imagenet_class_embedding + beta * adapter_output
     ```

### Key Components
- **Adapter:** The only trainable layer, adapting the text encoding to generate images conditioned on textual descriptions.
- **Hyperparameters:** `alpha` and `beta` control the weighting of the class embedding and adapter output.

## Future Improvements

- **Bounding Box Conditioning:** Incorporate bounding box conditioning similar to ControlNet, enabling the model to handle spatial constraints.
- **Hyperparameter Tuning:** Experiment with different values for `alpha` and `beta`, as well as additional hyperparameters.
- **LoRA Training for VAR:** Explore training the VAR using Low-Rank Adaptation (LoRA) for improved efficiency and fine-tuning capabilities.
- **Diverse Datasets:** Train on a more diverse dataset to enhance generalization and adaptability across different styles and categories.

## Acknowledgments
The architecture and model are inspired by the Visual Autoregressive (VAR) model from the paper [Visual Autoregressive Modeling](https://arxiv.org/abs/2404.02905). I recently came across [VAR-CLIP: Text-to-Image Generator with Visual Auto-Regressive Modeling](https://arxiv.org/abs/2408.01181), which integrates VAR with CLIP for text-to-image generation. For more detailed insights on VAR-CLIP’s contributions, please refer to the paper.

## License
This project is licensed under the MIT License.