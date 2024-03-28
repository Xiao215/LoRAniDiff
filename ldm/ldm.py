import torch
import torch.nn as nn
import torch.nn.functional as F
import ldm.utils.model_loader as model_loader
from ldm.model.clip import CLIP
from ldm.model.encoder import VAE_Encoder
from ldm.model.decoder import VAE_Decoder
from ldm.model.diffusion import Diffusion
from ldm.module.ddpm import DDPMSampler


class StableDiffusion(nn.Module):
    def __init__(self, device, alpha=0.5, model_file=None, n_inference_steps=50, seed=None, width=256, height=256):
        super(StableDiffusion, self).__init__()
        self.alpha = alpha
        self.device = device
        self.HEIGHT = height
        self.WIDTH = width
        self.LATENTS_WIDTH = self.WIDTH // 8
        self.LATENTS_HEIGHT = self.HEIGHT // 8
        self.n_inference_steps = n_inference_steps
        # Initialize your models here. If model_file is provided, load the weights.
        if model_file is not None:
            models = model_loader.preload_models_from_standard_weights(model_file, device)
            self.encoder = models['encoder'].to(device)
            self.decoder = models['decoder'].to(device)
            self.diffusion = models['diffusion'].to(device)
            self.clip = models['clip'].to(device)
        else:
            self.encoder = VAE_Encoder().to(device)
            self.decoder = VAE_Decoder().to(device)
            self.diffusion = Diffusion().to(device)
            self.clip = CLIP().to(device)
        generator = torch.Generator(device=device)
        if seed is None:
            generator.seed()
        else:
            generator.manual_seed(seed)

    def forward(self, images, captions, tokenizer, strength=0.8):
        batch_size = len(captions)
        latents_shape = (batch_size, 4, self.LATENTS_HEIGHT, self.LATENTS_WIDTH)
        generator = torch.Generator(device=self.device)
        tokens = tokenizer(captions, padding="max_length", max_length=77, return_tensors="pt", truncation=True).input_ids.to(self.device)
        # tokens = torch.tensor(tokens, dtype=torch.long, device=self.device)
        context = self.clip(tokens)
        sampler = DDPMSampler(generator)
        sampler.set_inference_timesteps(self.n_inference_steps)
        if images is not None:
            encoder_noise = torch.randn(latents_shape, generator=generator, device=self.device)
            latents = self.encoder(images, encoder_noise)
            sampler.set_strength(strength)
            latents = sampler.add_noise(latents, sampler.timesteps[0])
        else:
            latents = torch.randn(latents_shape, generator=generator, device=self.device)

        timesteps = sampler.timesteps
        for timestep in timesteps:
            time_embedding = self.get_time_embedding(timestep).to(self.device)
            model_input = latents
            model_output = self.diffusion(model_input, context, time_embedding)
            latents = sampler.step(timestep, latents, model_output)

        return self.rescale(self.decoder(latents), (-1, 1), (0, 1), clamp=True), context

    @staticmethod
    def get_time_embedding(timestep):
        # Shape: (160,)
        freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160)
        # Shape: (1, 160)
        x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]
        # Shape: (1, 160 * 2)
        return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)

    def compute_loss(self, generated_images, target_images, text_embeddings):
        rec_loss = F.mse_loss(generated_images, target_images)
        generated_images = (generated_images + 1) / 2.0
        generated_images = generated_images.clamp(0, 1)
        image_features = self.clip.encode_image(generated_images).float()
        image_features = F.normalize(image_features, dim=-1)
        text_embeddings = F.normalize(text_embeddings, dim=-1)
        similarity = torch.sum(image_features * text_embeddings, dim=-1)
        clip_loss = -torch.mean(similarity)
        total_loss = (1 - self.alpha) * rec_loss + self.alpha * clip_loss
        return total_loss

    def rescale(x, old_range, new_range, clamp=False):
        old_min, old_max = old_range
        new_min, new_max = new_range
        x -= old_min
        x *= (new_max - new_min) / (old_max - old_min)
        x += new_min
        if clamp:
            x = x.clamp(new_min, new_max)
        return x