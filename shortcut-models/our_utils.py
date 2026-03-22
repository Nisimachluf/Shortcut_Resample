import yaml
import torch
import scipy
from torch_model import DiT
from torch_vae import AutoencoderKL

import numpy as np
from PIL import Image
import torch.fft as fft
import torch.nn.functional as F
import torchvision.transforms as transforms

from sampling_tools import build_step_schedule

def load_models(yaml_path = "shortcutmodel_args.yaml"):
    device = "cuda"

    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
        
    dit_args = config['dit_args']
    dit_args["dtype"] = torch.float32
    vae_args = config['vae_args']
    dit_weights = config.get('dit_weights', None)
    vae_weights = config.get('vae_weights', None)

    dit = DiT(**dit_args).to(device)
    if dit_weights is not None:
        dit.load_state_dict(torch.load(dit_weights, map_location=device))
    dit.eval()
    vae = AutoencoderKL(**vae_args).to(device)
    if vae_weights is not None:
        vae.load_state_dict(torch.load(vae_weights, map_location=device))
    vae.eval()
    return dit, vae



def rescale_img(x):
    """Rescales image from [-1, 1], float to [0, 255] uint8

    Args:
        x (torch.Tensor or np.ndarray): Input image tensor or array.

    Returns:
        np.ndarray: Rescaled image in [0, 255] uint8 format.
    """
    if isinstance(x, torch.Tensor):
        x = np.array(x.detach().cpu().numpy())
    shape = x.shape
    if len(shape) == 4 and shape[1] == 3:
        # it batch and channels first, move channels to last
        x = x.transpose(0, 2, 3, 1)
    elif len(shape) == 3 and shape[0] == 3:
        # if channels first, move channels to last
        x = x.transpose(1, 2, 0)
    x = x * 0.5 + 0.5  # [-1,1] to [0,1]
    x = np.clip(x, 0, 1)
    x = (x * 255).astype(np.uint8)
    return x

def decode(x, vae, calc_grad=False, rescale=False):
    """Decodes an image using the VAE model.

    Args:
        x (torch.Tensor or np.ndarray): Input image tensor or array.
        vae (AutoencoderKL): VAE model for decoding.
        calc_grad (bool, optional): Whether to calculate gradients. Defaults to False.
        rescale (bool, optional): Whether to rescale the output image. Defaults to True.

    Returns:
        np.ndarray: Decoded image in [0, 255] uint8 format if rescale is True, else raw tensor.
    """
    if calc_grad:
        x_decoded = vae.decode(x)
    else:
        with torch.no_grad():
            x_decoded = vae.decode(x)
    if not rescale:
        return x_decoded
    return rescale_img(x_decoded)

def encode(img, vae, calc_grad=False, normalize=False):
    """Encodes an image using the VAE model.

    Args:
        img (torch.Tensor or np.ndarray): Input image tensor or array.
        vae (AutoencoderKL): VAE model for encoding.
        calc_grad (bool, optional): Whether to calculate gradients. Defaults to False.
        normalize (bool, optional): Whether to normalize the input image. Defaults to False.

    Returns:
        torch.Tensor: Encoded latent representation.
    """
    if isinstance(img, np.ndarray):
        img = torch.from_numpy(img).float().to(vae.device)
    if normalize:
        img = img / 255.0
        img = (img - 0.5)/0.5
    img = img.float()
    if calc_grad:
        lat = vae.encode(img)
    else:
        with torch.no_grad():
            lat = vae.encode(img)
    return lat


def stochastic_encoding(img, ti, dts, noise=None):
    """Computes the stochastic encoding of an image to time ti/dts using:
    x_t = t*img + (1-t)*noise, where t = ti/dts

    Args:
        img (torch.Tensor or np.ndarray): Input image tensor or array.
        ti (float): Current time step.
        dts (float): Total number of time steps.
        noise (torch.Tensor, optional): Noise tensor. Defaults to None.
    """
    if noise is None:
        noise = torch.randn_like(img)
    t = ti/dts
    return t*img + (1-t)*noise

def step_to_end(z, v, t):
    """Make a stop from z_t to z_1 using the velocity v_t and time t, where t is in [0, 1]"""
    return z + v*(1-t)

# def gaussian_kernel(kernel_size=61, sigma=3.0, channels=3, device='cuda'):
#     """
#     Create a 2D Gaussian kernel for convolution
#     """
#     ax = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1., device=device)
#     xx, yy = torch.meshgrid(ax, ax, indexing='ij')
#     kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
#     kernel = kernel / kernel.sum()
#     kernel = kernel.view(1, 1, kernel_size, kernel_size)
#     kernel = kernel.repeat(channels, 1, 1, 1)  # multi-channel
#     return kernel

def pad_and_shift_kernel(kernel, H, W):
    """
    Pads kernel to (H, W) centered, then ifftshifts it so that FFT interprets
    the kernel's center at the origin (0,0). This prevents circular shifts.
    """
    kh, kw = kernel.shape[-2:]

    pad_h = H - kh
    pad_w = W - kw

    # Centered padding
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    kernel_padded = F.pad(kernel, (pad_left, pad_right, pad_top, pad_bottom))

    # Move kernel center to (0,0) for FFT convolution
    kernel_shifted = fft.ifftshift(kernel_padded, dim=(-2, -1))

    return kernel_shifted


def solve_bp_fft_operator2(y, x0, kernel, rho=1.0, eps=1e-12):
    """
    Solves: x = (A^T A + rho I)^-1 (A^T y + rho x0)
    where A is convolution with kernel k.

    All FFT operations are done with correct kernel centering to avoid shifts.
    """

    B, C, H, W = y.shape

    # Prepare kernel
    if kernel.ndim == 2:  # (kh, kw)
        kernel = kernel.unsqueeze(0)  # -> (1, kh, kw)
    
    # Pad and shift kernel for FFT convolution
    kernel_reshaped = kernel.reshape(kernel.shape[0], 1, *kernel.shape[-2:])
    kernel_padded = pad_and_shift_kernel(kernel_reshaped, H, W)
    
    # Broadcast kernel if needed
    if kernel_padded.shape[0] == 1 and C > 1:
        kernel_padded = kernel_padded.repeat(C, 1, 1, 1)

    # Compute FFT of kernel (adjoint via conj)
    k_f = fft.rfft2(kernel_padded.reshape(C, H, W))   # (C, H, W/2+1)

    # Expand kernel to batch dimension
    k_f = k_f.unsqueeze(0).repeat(B, 1, 1, 1)         # (B,C,H,W/2+1)
    k_f = k_f.reshape(B * C, H, W // 2 + 1)

    # FFT of inputs
    y_f  = fft.rfft2(y.reshape(B * C, H, W))
    x0_f = fft.rfft2(x0.reshape(B * C, H, W))

    # Numerator: K* ⊙ Y + rho X0
    numerator = torch.conj(k_f) * y_f + rho * x0_f

    # Denominator: |K|^2 + rho
    denominator = (k_f.abs() ** 2) + rho + eps

    # Solve per-frequency
    x_f = numerator / denominator

    # Back to spatial domain
    x = fft.irfft2(x_f, s=(H, W))
    x = x.reshape(B, C, H, W)

    return x

def apply_convolution_operator(image, kernel):
    """
    Apply convolution operator A to an image using FFT: y = A * x
    
    Parameters:
    -----------
    image : torch.Tensor
        Input image of shape (B, C, H, W) or (C, H, W) or (H, W)
    kernel : torch.Tensor
        Convolution kernel of shape (kh, kw) or (C, kh, kw)
        
    Returns:
    --------
    torch.Tensor
        Convolved image, same shape as input
    """
    # Handle different input shapes
    original_shape = image.shape
    
    if image.ndim == 2:  # (H, W)
        image = image.unsqueeze(0).unsqueeze(0)  # -> (1, 1, H, W)
    elif image.ndim == 3:  # (C, H, W)
        image = image.unsqueeze(0)  # -> (1, C, H, W)
    
    B, C, H, W = image.shape
    
    # Prepare kernel
    if kernel.ndim == 2:  # (kh, kw)
        kernel = kernel.unsqueeze(0)  # -> (1, kh, kw)
    
    # Pad and shift kernel for FFT convolution
    kernel_reshaped = kernel.reshape(kernel.shape[0], 1, *kernel.shape[-2:])
    kernel_padded = pad_and_shift_kernel(kernel_reshaped, H, W)
    
    # Broadcast kernel if needed
    if kernel_padded.shape[0] == 1 and C > 1:
        kernel_padded = kernel_padded.repeat(C, 1, 1, 1)
    
    # Compute FFT of kernel
    k_f = fft.rfft2(kernel_padded.reshape(C, H, W))
    
    # Expand to batch dimension
    k_f = k_f.unsqueeze(0).repeat(B, 1, 1, 1)  # (B, C, H, W/2+1)
    k_f = k_f.reshape(B * C, H, W // 2 + 1)
    
    # FFT of image
    x_f = fft.rfft2(image.reshape(B * C, H, W))
    
    # Apply convolution in frequency domain
    y_f = k_f * x_f
    
    # Back to spatial domain
    y = fft.irfft2(y_f, s=(H, W))
    y = y.reshape(B, C, H, W)
    
    # Restore original shape
    if len(original_shape) == 2:
        y = y.squeeze(0).squeeze(0)
    elif len(original_shape) == 3:
        y = y.squeeze(0)
    
    return y


def read_image_to_tensor(image_path, normalize=True, device='cpu', return_np=False):
    """
    Read an image file and convert it to a PyTorch tensor.
    
    Args:
        image_path (str): Path to the image file
        normalize (bool): If True, normalize pixel values to [-1, 1]. Default: True
        device (str): Device to place the tensor on ('cpu' or 'cuda'). Default: 'cpu'
    
    Returns:
        torch.Tensor: Image tensor of shape (C, H, W) with values in [0, 1] if normalized,
                     or [0, 255] if not normalized
    
    Example:
        >>> img_tensor = read_image_to_tensor('path/to/image.jpg')
        >>> print(img_tensor.shape)  # (3, H, W)
    """
    # Read image using PIL
    image = Image.open(image_path).convert('RGB')
    
    # Convert to tensor
    if normalize:
        # Normalize to [-1, 1]
        transform = transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        img_tensor = transform(image)
    else:
        # Keep as [0, 255]
        img_array = np.array(image)
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float()
    
    # Move to specified device
    img_tensor = img_tensor.to(device)
    
    if not return_np:
        return img_tensor[None, ...]
    return img_tensor[None, ...], np.asarray(image)

def to_numpy(t):
    """Convert a torch tensor to a numpy ndarray

    Args:
        t (torch.Tensor): Input image tensor.

    Returns:
        np.ndarray: Image array.
    """
    if t.shape[0] == 1:
        t = t[0]
    if len(t.shape) == 3:
        t = t.permute(1, 2, 0)
    else:
        t = t.permute(0, 2, 3, 1)
    n = torch.clamp((t * 0.5 + 0.5) * 255, 0, 255)
    n = n.detach().cpu().numpy().astype("uint8")
    return n

def calc_t(ti, dts):
    t = ti / dts
    return t

def calc_v(model, x, t, denoise_timesteps, device="cuda", cfg_scale=0, num_classes=1):
    labels = torch.randint(0, num_classes, (x.shape[0],), device=device)
    batch_size = x.shape[0]

    t_vector = torch.full((batch_size,), t, dtype=torch.float32, device=device)
    dt_flow = int(np.log2(denoise_timesteps))
    dt_base = torch.ones(batch_size, dtype=torch.float32, device=device) * dt_flow
    # Classifier-free guidance
    labels_uncond = torch.ones_like(labels) * num_classes
    if cfg_scale == 1:
        v, *_ = model(x, t_vector, dt_base, labels)
    elif cfg_scale == 0:
        v, *_ = model(x, t_vector, dt_base, labels_uncond)
    else:
        v_pred_uncond, *_ = model(x, t_vector, dt_base, labels_uncond)
        v_pred_label, *_ = model(x, t_vector, dt_base, labels)
        v = v_pred_uncond + cfg_scale * (v_pred_label - v_pred_uncond)
    return v



            
class GaussianNoise:
    def __init__(self, sigma):
        self.sigma = sigma
    
    def __call__(self, data):
        return data + torch.randn_like(data, device=data.device) * self.sigma
    
def get_debluring_tools(kernel_size=61, std=3.0, noise_sigma=0.05, device="cuda"):
    n = np.zeros((kernel_size, kernel_size))
    n[kernel_size // 2,kernel_size // 2] = 1
    k = scipy.ndimage.gaussian_filter(n, sigma=std)
    try:
        k = torch.from_numpy(k)
    except:
        k = torch.as_tensor(k)
    k = k.to(device)
            
    return k.float(), GaussianNoise(noise_sigma)

def build_regularly_decaying_schedule(dts):
    if dts == 128:
        return build_step_schedule([1.0], [dts])
    fracs = [(dts-1)/dts]
    step_sizes = [dts]
    while dts < 128:
        dts*=2
        if dts < 128:
            fracs.append(1/dts)
            step_sizes.append(dts)
        else:
            fracs.append(2/128)
            step_sizes.append(128)
        
    return build_step_schedule(fracs, step_sizes)
    