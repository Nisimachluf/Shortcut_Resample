import torch
import torch.nn as nn
from torch_model import DiT
from torch_vae import AutoencoderKL

import yaml

class ShortcutModel(nn.Module):
    def __init__(self, yaml_path, device='cuda'):
        super().__init__()
        self.device = device
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        dit_args = config['dit_args']
        dit_args["device"] = device
        if isinstance(dit_args["dtype"], str):
            dit_args["dtype"] = eval(f'torch.{dit_args["dtype"]}')
            
        vae_args = config['vae_args']
        dit_weights = config.get('dit_weights', None)
        vae_weights = config.get('vae_weights', None)

        self.dit = DiT(**dit_args).to(device)
        if dit_weights is not None:
            self.dit.load_state_dict(torch.load(dit_weights, map_location=device))
        self.vae = AutoencoderKL(**vae_args).to(device)
        if vae_weights is not None:
            self.vae.load_state_dict(torch.load(vae_weights, map_location=device))

    def apply_model(self, x, t, dt, labels):
        """
        Apply DiT model on input x, timestep t, dt, and labels.
        """
        return self.dit(x, t, dt, labels)

    def encode_first_stage(self, img):
        """
        Encode input image to latent space using VAE encoder.
        """
        return self.vae.encode(img)

    @torch.no_grad()
    def decode_first_stage(self, z):
        """
        Decode latent z to image using VAE decoder (no grad).
        """
        # z= torch.permute(z, (0, 3, 1, 2))
        return self.vae.decode(z)

    def differentiable_decode_first_stage(self, z):
        """
        Decode latent z to image using VAE decoder (with grad).
        """
        # z= torch.permute(z, (0, 3, 1, 2))
        return self.vae.decode(z)