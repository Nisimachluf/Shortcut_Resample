import torch
import torch.nn.functional as F

def gaussian_kernel(kernel_size=61, sigma=3.0, channels=3, device='cuda'):
    """
    Create a 2D Gaussian kernel for convolution
    """
    ax = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1., device=device)
    xx, yy = torch.meshgrid(ax, ax, indexing='ij')
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, kernel_size, kernel_size)
    kernel = kernel.repeat(channels, 1, 1, 1)  # multi-channel
    return kernel

# A(x)
def blur_operator(x):
    kernel = gaussian_kernel()
    """
    Forward operator: Gaussian blur via convolution
    x: [B, C, H, W]
    kernel: [C, 1, kH, kW]
    """
    padding = kernel.shape[-1] // 2
    return F.conv2d(x, kernel, padding=padding, groups=x.shape[1])

# A^T(y)
def blur_adjoint(y):
    """
    Adjoint operator: convolution with flipped kernel
    y: [B, C, H, W]
    kernel: [C, 1, kH, kW]
    """
    kernel = gaussian_kernel()
    kernel_flip = torch.flip(kernel, [2, 3])
    padding = kernel.shape[-1] // 2
    return F.conv2d(y, kernel_flip, padding=padding, groups=y.shape[1])

def laplacian(x):
    """Compute Laplacian for smoothing."""
    x = x.float()  # ensure float32
    # make kernel float32 and on same device as x
    kernel = torch.tensor([[[[0,1,0],[1,-4,1],[0,1,0]]]], dtype=torch.float32, device=x.device)
    kernel = kernel.repeat(x.shape[1],1,1,1)
    return F.conv2d(x, kernel, padding=1, groups=x.shape[1])


