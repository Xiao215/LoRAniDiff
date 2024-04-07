![Python](https://img.shields.io/badge/python-3.10-blue.svg)
![Repo Size](https://img.shields.io/github/repo-size/Sulstice/global-chem)
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FXiao215%2FLoRAniDiff&count_bg=%23BD7EF0&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)


# LoRAniDiff

LoRAniDiff is an innovative image generation project leveraging the power of table diffusion models, fine-tuned with LoRA on a curated set of approximately 1,000 Pixiv images. This project combines the strengths of diffusion models with Low-Rank Adaptation (LoRA) to offer enhanced control and creativity in generating detailed and expressive imagery.

## Getting Started

### Prerequisites

Before you begin, please note that the `requirements.txt` for this project is still under preparation and might not cover all dependencies correctly, which could result in installation failures.

### Installation

1. Clone this repository to your local machine using:
   ```bash
   git clone git@github.com:Xiao215/LoRAniDiff.git
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
> **Note:** As mentioned, the `requirements.txt` is not finalized yet, so installation may fail.

### Obtaining the Model Weights

To use LoRAniDiff, you'll need to obtain the model weights by running:
```bash
python3 ldm/utils/get_weight.py
```
### Using the Model

With the model weights obtained, you can start generating images as follows:

```python
from transformers import CLIPTokenizer
from ldm.ldm import LoRAniDiff
import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = CLIPTokenizer("model_weight/vocab.json", merges_file="model_weight/merges.txt")
pt_file = "model_weight/LoRAniDiff.pt"
prompt = "give me an image of a cat with a hat"

model = LoRAniDiff(device=DEVICE, seed=42, tokenizer=tokenizer)
model.load_state_dict(torch.load(pt_file, map_location=DEVICE))

output_image = model.generate(prompt, input_image=None) # Specify your input_image if available
```

### Obtaining the Datasets

This project provides two datasets for experimentation: TextCaps and Pixiv. The Pixiv dataset was manually scrapped and labeled by Llava7B. To obtain them, run:
```python
python3 ldm/dataset/get_data.py
```

## Citation

This project is inspired by and based upon the work described in the paper "High-Resolution Image Synthesis with Latent Diffusion Models". We extend our gratitude to the authors for their groundbreaking contributions to the field:
```
@misc{rombach2021highresolution,
title={High-Resolution Image Synthesis with Latent Diffusion Models},
author={Robin Rombach and Andreas Blattmann and Dominik Lorenz and Patrick Esser and Björn Ommer},
year={2021},
eprint={2112.10752},
archivePrefix={arXiv},
primaryClass={cs.CV}
}
```



Special thanks to [pytorch-stable-diffusion](https://github.com/hkproj/pytorch-stable-diffusion) for providing valuable resources and support in building our stable diffusion model.

## Disclaimer

Please note that LoRAniDiff is designed for experimental and fun purposes only. It should not be used for any other purposes.
