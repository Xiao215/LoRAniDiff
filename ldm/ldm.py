import torch
import torch.nn as nn
import torch.nn.functional as F
import ldm.utils.model_loader as model_loader
from ldm.model.clip import CLIP
from ldm.model.encoder import VAE_Encoder
from ldm.model.decoder import VAE_Decoder
from ldm.model.diffusion import Diffusion
from ldm.module.ddpm import DDPMSampler
from PIL import Image


class LoRAniDiff(nn.Module):
    def __init__(
        self,
        tokenizer,
        device,
        alpha=0.5,
        model_file=None,
        n_inference_steps=50,
        seed=None,
        width=512,
        height=512,
    ):
        super(LoRAniDiff, self).__init__()
        self.alpha = alpha
        self.device = device
        self.HEIGHT = height
        self.WIDTH = width
        self.LATENTS_WIDTH = self.WIDTH // 8
        self.LATENTS_HEIGHT = self.HEIGHT // 8
        self.n_inference_steps = n_inference_steps
        # Initialize your models here. If model_file is provided, load the weights.
        # NOTE: model_file should be None and is deprecated.
        if model_file is not None:
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
        if seed is None:
            self.generator.seed()
        else:
            self.generator.manual_seed(seed)
        self.tokenizer = tokenizer

    def forward(
        self,
        prompt,
        uncond_prompt,
        images=None,
        do_cfg=True,
        cfg_scale=7.5,
        strength=0.8,
    ):
        batch_size = len(prompt)
        # Assume that the unconditional prompt is the same for all samples in
        # the batch, which is empty
        latents_shape = (
            batch_size,
            4,
            self.LATENTS_HEIGHT,
            self.LATENTS_WIDTH)

        context = None
        if do_cfg:
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
            time_embedding = LoRAniDiff.get_time_embedding(
                timestep).to(self.device)
            model_input = latents
            if do_cfg:
                model_input = model_input.repeat(2, 1, 1, 1)
            model_output = self.diffusion(model_input, context, time_embedding)
            if do_cfg:
                output_cond, output_uncond = model_output.chunk(2)
                model_output = cfg_scale * \
                    (output_cond - output_uncond) + output_uncond
            latents = sampler.step(timestep, latents, model_output)
        print(f"latents: {latents}")
        print(f"decode latent: {self.decoder(latents)}")
        images = LoRAniDiff.rescale(
            self.decoder(latents), (-1, 1), (0, 255), clamp=True
        )
        images = images.permute(0, 2, 3, 1)
        images = images.to("cpu", torch.uint8).numpy()
        print(f"image: {images[0]}")
        return images[0], context, uncond_context

    @staticmethod
    def get_time_embedding(timestep):
        # Shape: (160,)
        freqs = torch.pow(
            10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160
        )
        # Shape: (1, 160)
        x = torch.tensor(
            [timestep], dtype=torch.float32)[
            :, None] * freqs[None]
        # Shape: (1, 160 * 2)
        return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)

    @staticmethod
    def rescale(x, old_range, new_range, clamp=False):
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
        generated_images,
        target_images,
        text_embeddings_cond,
        text_embeddings_uncond,
        cfg_scale,
    ):
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

        clip_loss = clip_loss_cond - cfg_scale * \
            (clip_loss_cond - clip_loss_uncond)

        total_loss = (1 - self.alpha) * rec_loss + self.alpha * clip_loss
        return total_loss

    def generate(self, caption, input_image=None, strength=0.8):
        """
        Generate an image based on a caption, and optionally, an initial image.

        Parameters:
        - caption: str, the caption based on which to generate an image.
        - initial_image: torch.Tensor, optional initial image for image-to-image generation.
        - strength: float, the strength of the modification for image-to-image generation.

        Returns:
        - generated_image: torch.Tensor, the generated image tensor.
        - context: torch.Tensor, the context tensor from the CLIP model.
        """
        self.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Ensure no gradients are calculated
            captions = [caption]  # Encapsulate the caption in a list
            if input_image is not None:
                if len(
                        input_image.shape) == 3:  # If single image, add batch dimension
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
            )
            image = Image.fromarray(generated_image)
            print(f"Image generated successfully.")
            print(f"Image size: {image.size}")
            print(f"image: {image}")

        return image
