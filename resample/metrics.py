import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim

class PSNR:

    def __init__(self):
        self.sum = 0.0
        self.count = 0
        self.mean = None

    def __call__(self, img_gt, img_test):
        """
        img_gt, img_test: np.ndarray or torch.Tensor, shape (H, W, C) or (C, H, W), range [0, 1] or [0, 255]
        """
        if isinstance(img_gt, torch.Tensor):
            img_gt = img_gt.detach().cpu().numpy()
        if isinstance(img_test, torch.Tensor):
            img_test = img_test.detach().cpu().numpy()
        if img_gt.max() > 1.1:
            img_gt = img_gt / 255.0
        if img_test.max() > 1.1:
            img_test = img_test / 255.0
        if img_gt.shape != img_test.shape:
            raise ValueError("Input images must have the same shape.")
        value = sk_psnr(img_gt, img_test, data_range=1.0)
        self.sum += value
        self.count += 1
        self.mean = self.sum / self.count
        return value

class SSIM:

    def __init__(self):
        self.sum = 0.0
        self.count = 0
        self.mean = None

    def __call__(self, img_gt, img_test):
        """
        img_gt, img_test: np.ndarray or torch.Tensor, shape (H, W, C) or (C, H, W), range [0, 1] or [0, 255]
        """
        if isinstance(img_gt, torch.Tensor):
            img_gt = img_gt.detach().cpu().numpy()
        if isinstance(img_test, torch.Tensor):
            img_test = img_test.detach().cpu().numpy()
        if img_gt.max() > 1.1:
            img_gt = img_gt / 255.0
        if img_test.max() > 1.1:
            img_test = img_test / 255.0
        if img_gt.shape != img_test.shape:
            raise ValueError("Input images must have the same shape.")
        multichannel = img_gt.ndim == 3 and img_gt.shape[-1] in [3, 4]
        value = sk_ssim(img_gt, img_test, data_range=1.0, channel_axis=-1 if multichannel else None)
        self.sum += value
        self.count += 1
        self.mean = self.sum / self.count
        return value

class LPIPS:

    def __init__(self, net='alex', device='cuda'):
        import lpips
        self.model = lpips.LPIPS(net=net).to(device)
        self.device = device
        self.sum = 0.0
        self.count = 0
        self.mean = None

    def __call__(self, img_gt, img_test):
        """
        img_gt, img_test: np.ndarray or torch.Tensor, shape (H, W, C) or (C, H, W), range [0, 1] or [0, 255]
        """
        if isinstance(img_gt, np.ndarray):
            img_gt = torch.from_numpy(img_gt)
        if isinstance(img_test, np.ndarray):
            img_test = torch.from_numpy(img_test)
        if img_gt.max() > 1.1:
            img_gt = img_gt / 255.0
        if img_test.max() > 1.1:
            img_test = img_test / 255.0
        # LPIPS expects (N, 3, H, W) in [-1, 1]
        if img_gt.ndim == 3:
            if img_gt.shape[0] == 3:
                pass  # (3, H, W)
            elif img_gt.shape[-1] == 3:
                img_gt = img_gt.permute(2, 0, 1)
            else:
                raise ValueError("Input image must have 3 channels.")
        if img_test.ndim == 3:
            if img_test.shape[0] == 3:
                pass
            elif img_test.shape[-1] == 3:
                img_test = img_test.permute(2, 0, 1)
            else:
                raise ValueError("Input image must have 3 channels.")
        img_gt = img_gt.unsqueeze(0).to(self.device)
        img_test = img_test.unsqueeze(0).to(self.device)
        img_gt = img_gt * 2 - 1
        img_test = img_test * 2 - 1
        with torch.no_grad():
            dist = self.model(img_gt, img_test)
        value = dist.item()
        self.sum += value
        self.count += 1
        self.mean = self.sum / self.count
        return value
