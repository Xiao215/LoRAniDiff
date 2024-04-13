"""
This module implements the LoRAniDiff model for latent diffusion, which can be used for
text-to-image generation. It integrates components such as VAE encoder and decoder, CLIP model
for text and image embeddings, and a diffusion model for the generative process.
"""

import os
import torch
from torch import nn
import torch.nn.functional as F
from PIL import Image
from transformers import PreTrainedTokenizer
from ldm.utils import model_loader
from ldm.model.clip import CLIP
from ldm.model.encoder import VAE_Encoder
from ldm.model.decoder import VAE_Decoder
from ldm.model.diffusion import Diffusion
from ldm.module.ddpm import DDPMSampler


class LoRAniDiff(nn.Module):
    """
    LoRAniDiff model integrates various components for generating images from text prompts.
    It can be conditioned on text input to generate relevant images or perform modifications
    on existing images to align them with given text prompts.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        device: torch.device,
        alpha: float = 0.5,
        model_file: str | None = None,
        n_inference_steps: int = 50,
        seed: int | None = None,
        width: int = 512,
        height: int = 512,
    ):
        super().__init__()
        self.alpha = alpha
        self.device = device
        self.height = height
        self.width = width
        self.latents_width = self.width // 8
        self.latents_height = self.height // 8
        self.n_inference_steps = n_inference_steps

        if model_file is not None:
            # Assuming model_loader can preload models given a file
            models = model_loader.preload_models_from_standard_weights(
                model_file, device
            )
            self.encoder = models["encoder"].to(device)
            self.decoder = models["decoder"].to(device)
            self.diffusion = models["diffusion"].to(device)
            self.clip = models["clip"].to(device)
        else:
            self.encoder = VAE_Encoder().to(device)
            self.decoder = VAE_Decoder().to(device)
            self.diffusion = Diffusion().to(device)
            self.clip = CLIP().to(device)

        self.generator = torch.Generator(device=device)
        if seed is not None:
            self.generator.manual_seed(seed)
        self.tokenizer = tokenizer

    def forward(
        self,
        prompt: list[str],
        uncond_prompt: list[str] = [""],
        images: torch.Tensor | None = None,
        do_cfg: bool = True,
        cfg_scale: float = 7.5,
        strength: float = 0.8,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the model to generate images based on text prompts and/or initial images.

        Parameters:
        - prompt (list[str]): Text prompts for conditional generation.
        - uncond_prompt (list[str]): Text prompts for unconditional generation, usually empty.
        - images (torch.Tensor | None): Optional initial images for image-to-image generation.
        - do_cfg (bool): Whether to apply classifier-free guidance.
        - cfg_scale (float): Scale for classifier-free guidance.
        - strength (float): Strength of modifications for image-to-image generation.

        Returns:
        - tuple containing generated images, conditional context, and unconditional context.
        """
        batch_size = len(prompt)
        # Assume that the unconditional prompt is the same for all samples in
        # the batch, which is empty
        latents_shape = (batch_size, 4, self.latents_height, self.latents_width)

        context = None
        if do_cfg:
            print(prompt)
            cond_tokens = self.tokenizer.batch_encode_plus(
                prompt, padding="max_length", max_length=77
            ).input_ids
            cond_tokens = torch.tensor(
                cond_tokens, dtype=torch.long, device=self.device
            )
            cond_context = self.clip(cond_tokens)

            uncond_tokens = self.tokenizer.batch_encode_plus(
                uncond_prompt, padding="max_length", max_length=77
            ).input_ids
            uncond_tokens = torch.tensor(
                uncond_tokens, dtype=torch.long, device=self.device
            )
            uncond_context = self.clip(uncond_tokens)
            context = torch.cat([cond_context, uncond_context])
        else:
            tokens = self.tokenizer.batch_encode_plus(
                prompt, padding="max_length", max_length=77
            ).input_ids
            tokens = torch.tensor(tokens, dtype=torch.long, device=self.device)
            context = self.clip(tokens)

        sampler = DDPMSampler(self.generator)
        sampler.set_inference_timesteps(self.n_inference_steps)

        latents = None
        if images is not None:
            input_image_tensor = LoRAniDiff.rescale(input_image_tensor, (0, 255), (-1, 1))
            # (Height, Width, Channel) -> (Batch_Size, Height, Width, Channel)
            input_image_tensor = input_image_tensor.unsqueeze(0)
            # (Batch_Size, Height, Width, Channel) -> (Batch_Size, Channel, Height, Width)
            input_image_tensor = input_image_tensor.permute(0, 3, 1, 2)

            encoder_noise = torch.randn(
                latents_shape, generator=self.generator, device=self.device
            )
            latents = self.encoder(images, encoder_noise)
            sampler.set_strength(strength)
            latents = sampler.add_noise(latents, sampler.timesteps[0])
        else:
            latents = torch.randn(
                latents_shape, generator=self.generator, device=self.device
            )

        for timestep in sampler.timesteps:
            time_embedding = LoRAniDiff.get_time_embedding(timestep).to(self.device)
            model_input = latents
            if do_cfg:
                model_input = model_input.repeat(2, 1, 1, 1)
            model_output = self.diffusion(model_input, context, time_embedding)
            if do_cfg:
                output_cond, output_uncond = model_output.chunk(2)
                model_output = cfg_scale * (output_cond - output_uncond) + output_uncond
            latents = sampler.step(timestep, latents, model_output)
        images = self.decoder(latents)
        images = LoRAniDiff.rescale(images, (-1, 1), (0, 255), clamp=True)
        images = images.permute(0, 2, 3, 1)
        images = images.to("cpu", torch.uint8).numpy()
        return images[0], context, uncond_context

    @staticmethod
    def get_time_embedding(timestep: int) -> torch.Tensor:
        # Shape: (160,)
        freqs = torch.pow(
            10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160
        )
        # Shape: (1, 160)
        x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]
        # Shape: (1, 160 * 2)
        return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)

    @staticmethod
    def rescale(
        x: torch.Tensor,
        old_range: tuple[float, float],
        new_range: tuple[float, float],
        clamp: bool = False,
    ) -> torch.Tensor:
        old_min, old_max = old_range
        new_min, new_max = new_range
        x -= old_min
        x *= (new_max - new_min) / (old_max - old_min)
        x += new_min
        if clamp:
            x = x.clamp(new_min, new_max)
        return x

    def compute_loss(
        self,
        generated_images: torch.Tensor,
        target_images: torch.Tensor,
        text_embeddings_cond: torch.Tensor,
        text_embeddings_uncond: torch.Tensor,
        cfg_scale: float,
    ) -> torch.Tensor:
        rec_loss = F.mse_loss(generated_images, target_images)
        generated_images = (generated_images + 1) / 2.0
        generated_images = generated_images.clamp(0, 1)

        image_features = self.clip.encode_image(generated_images).float()
        image_features = F.normalize(image_features, dim=-1)
        text_features_cond = F.normalize(text_embeddings_cond, dim=-1)
        text_features_uncond = F.normalize(text_embeddings_uncond, dim=-1)

        clip_loss_cond = -torch.mean(
            torch.sum(image_features * text_features_cond, dim=-1)
        )
        clip_loss_uncond = -torch.mean(
            torch.sum(image_features * text_features_uncond, dim=-1)
        )

        clip_loss = clip_loss_cond - cfg_scale * (clip_loss_cond - clip_loss_uncond)

        total_loss = (1 - self.alpha) * rec_loss + self.alpha * clip_loss
        return total_loss

    def generate(
        self,
        caption: str,
        input_image: torch.Tensor | None = None,
        strength: float = 0.8,
        cfg_scale: float = 7.5,
    ) -> Image.Image:
        """
        Generates an image based on the provided text caption and optional input image for
        image-to-image generation.

        Parameters:
        - caption (str): The caption describing the desired output image.
        - input_image (torch.Tensor | None): Optional initial image for image-to-image generation.
        - strength (float): The strength of the transformation for image-to-image generation.

        Returns:
        - Image.Image: The generated image.
        """
        self.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Ensure no gradients are calculated
            captions = [caption]  # Encapsulate the caption in a list
            if input_image is not None:
                if len(input_image.shape) == 3:  # If single image, add batch dimension
                    input_image = input_image.unsqueeze(0)
                input_image = input_image.to(
                    self.device
                )  # Ensure the image is on the correct device
            # Generate the image
            generated_image, _, _ = self.forward(
                uncond_prompt=[""],
                images=input_image,
                prompt=captions,
                strength=strength,
                cfg_scale=cfg_scale,
            )
            image = Image.fromarray(generated_image)
        return image
