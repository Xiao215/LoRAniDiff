# Stable Diffusion Project - ECE324

## About

This repository hosts the Stable Diffusion project, a part of the ECE324: Machine Intelligence, Software, and Neural Networks course at the University of Toronto's Engineering Science program. The project focuses on implementing a high-resolution image synthesis model using PyTorch, based on the principles of Stable Diffusion models.

Stable Diffusion models represent an advanced area in machine learning, particularly within neural networks and image processing. These models are capable of generating detailed and coherent images from textual descriptions, showcasing the powerful capabilities of neural networks in the domain of image synthesis.

This implementation aims to explore and demonstrate the intersection of machine intelligence and software engineering, underlining the practical applications and impact of neural networks in image processing and generative models.

## Setup Guide

### Prerequisites

- Anaconda or Miniconda (Python 3.11)
- GPU access is recommended for efficient model training and inference

### Creating a Conda Environment

To set up the project environment, follow these steps:

1. **Clone the Repository:**

   ```bash
   git clone git@github.com:Xiao215/Stable-Diffusion.git
   cd Stable-Diffusion
   ```

2. **Create the Conda Environment:**

   An `environment.yml` file is provided to ease the setup of the necessary dependencies. Create the Conda environment by running:

   ```bash
   conda env create -f environment.yml
   ```

   This command creates a new environment named `ece324`, based on the specifications in the `environment.yml` file.

3. **Activate the Conda Environment:**

   Once the environment is created, activate it using:

   ```bash
   conda activate ece324
   ```

   Your environment is now set up with all the required dependencies.

## Citation

This project is based on the work described in the following paper:

```bibtex
@misc{rombach2021highresolution,
  title={High-Resolution Image Synthesis with Latent Diffusion Models},
  author={Robin Rombach and Andreas Blattmann and Dominik Lorenz and Patrick Esser and Björn Ommer},
  year={2021},
  eprint={2112.10752},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
```

**Note**: Verify that the environment name and other specific details in the `environment.yml` file match your project setup.
