import torch

from ldm.model.clip import CLIP
from ldm.model.encoder import VAE_Encoder
from ldm.model.decoder import VAE_Decoder
from ldm.model.diffusion import Diffusion
import ldm.utils.model_converter as model_converter


def preload_models_from_standard_weights(
        ckpt_path: str, device: torch.device,):
    state_dict = model_converter.load_from_standard_weights(ckpt_path, device)

    encoder = VAE_Encoder().to(device)
    encoder.load_state_dict(state_dict["encoder"], strict=True)

    decoder = VAE_Decoder().to(device)
    decoder.load_state_dict(state_dict["decoder"], strict=True)

    diffusion = Diffusion().to(device)
    diffusion.load_state_dict(state_dict["diffusion"], strict=True)

    clip = CLIP().to(device)
    clip.load_state_dict(state_dict["clip"], strict=True)

    return {
        "clip": clip,
        "encoder": encoder,
        "decoder": decoder,
        "diffusion": diffusion,
    }
