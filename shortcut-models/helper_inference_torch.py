
from time import sleep
from tqdm import tqdm
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"        
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"
import torch
import numpy as np
import jax
import jax.numpy as jnp
from torch_model import DiT
from utils.stable_vae import StableVAE
import matplotlib.pyplot as plt

def torch_inference(
    model_ckpt_path,
    out_dir,
    batch_size=32,
    image_size=32,
    in_channels=4,
    denoise_timesteps=128,
    num_generations=64,
    cfg_scale=1.0,
    num_classes=1,
    device="cuda",
    fid_stats=None
):
    # ----------- Load Torch DiT Model -----------
    torch_dit_args = {
        'patch_embed_in_channels': in_channels,
        'patch_size': 2,
        'hidden_size': 768,
        'depth': 12,
        'num_heads': 12,
        'mlp_ratio': 4,
        'out_channels': 4,
        'class_dropout_prob': 1,
        'num_classes': num_classes,
        'dropout': 0.0,
        'ignore_dt': False,
        'dtype': torch.float32,
        'device': device,
    }
    model = DiT(**torch_dit_args).to(device)
    model.load_state_dict(torch.load(model_ckpt_path, map_location=device))
    model.eval()
    print("Torch DiT loaded")
    # ----------- Initialize VAE (JAX) -----------
    vae = StableVAE.create()
    vae_decode = jax.jit(vae.decode)
    vae_rng = jax.random.PRNGKey(42)
    print("VAE initialized")
    # ----------- Optionally Load FID Network -----------
    if fid_stats is not None:
        from utils.fid import get_fid_network, fid_from_stats
        get_fid_activations = get_fid_network()
        truth_fid_stats = np.load(fid_stats)
        print("FID network loaded")
        # sleep(30)
    else:
        get_fid_activations = None
        truth_fid_stats = None

    # ----------- Inference Loop -----------
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
    all_images = []
    with torch.no_grad():
        for gen_idx in tqdm(range(num_generations // batch_size)):
            # Sample initial noise in latent space
            x = np.random.randn(batch_size, image_size, image_size, in_channels).astype(np.float32)
            x = torch.Tensor(x).to(device)
            labels = torch.randint(0, num_classes, (batch_size,), device=device)
            delta_t = 1.0 / denoise_timesteps

            for ti in tqdm(range(denoise_timesteps)):
                t = ti / denoise_timesteps
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

                # Euler sampling
                x = x + v * delta_t

            # VAE decode (JAX)
            x_np = x.detach().cpu().numpy()
            x_jax = jnp.array(x_np)
            x_decoded_jax = vae_decode(x_jax)
            x_decoded = np.array(x_decoded_jax)
            all_images.append(x_decoded)
            if out_dir is not None:
                x_decoded = x_decoded * 0.5 + 0.5  # [-1,1] to [0,1]
                x_decoded = np.clip(x_decoded, 0, 1)
                x_img = (x_decoded * 255).astype(np.uint8)
                for i in range(batch_size):
                    plt.imsave(os.path.join(out_dir, f"res_{gen_idx*batch_size+i}.png"), x_img[i])

    all_images = np.concatenate(all_images, axis=0)
    if out_dir is not None:
        np.save(os.path.join(out_dir, "generated_images.npy"), all_images)
        print(f"Saved {all_images.shape[0]} images to {out_dir}/generated_images.npy")

    # ----------- FID Calculation -----------
    if get_fid_activations is not None and truth_fid_stats is not None:
        # Resize images to 299x299x3 for FID
        from skimage.transform import resize
        resized_images = np.stack([resize(img, (299, 299, 3), anti_aliasing=True) for img in all_images])
        resized_images = np.clip(resized_images, -1, 1)
        acts = get_fid_activations(resized_images)
        if acts.ndim == 4 and acts.shape[1:3] == (1, 1):
            acts = acts[..., 0, 0, :] if acts.shape[-1] == 2048 else acts.squeeze((1,2))
        acts = np.array(acts)
        mu1 = np.mean(acts, axis=0)
        sigma1 = np.cov(acts, rowvar=False)
        if out_dir is not None:
            np.savez(f"{out_dir}/fid_stats.npz", mu=mu1, sigma=sigma1)
        
        fid = fid_from_stats(mu1, sigma1, truth_fid_stats['mu'], truth_fid_stats['sigma'])
        print(f"FID is {fid}")

if __name__ == "__main__":
    import argparse

    def get_args():
        parser = argparse.ArgumentParser(description="Torch DiT Inference with optional FID calculation")
        parser.add_argument('--model_ckpt_path', type=str, default="checkpoints/dit_pytorch_checkpoint.pth", help='Path to Torch DiT checkpoint')
        parser.add_argument('--out_dir', type=str, default=None, help='Output directory for images and npy')
        parser.add_argument('--batch_size', type=int, default=32, help='Batch size for inference')
        parser.add_argument('--image_size', type=int, default=32, help='Image size')
        parser.add_argument('--in_channels', type=int, default=4, help='Number of input channels')
        parser.add_argument('--denoise_timesteps', type=int, default=128, help='Number of denoising timesteps')
        parser.add_argument('--num_generations', type=int, default=512, help='Total number of images to generate')
        parser.add_argument('--cfg_scale', type=float, default=0, help='CFG scale')
        parser.add_argument('--num_classes', type=int, default=1, help='Number of classes')
        parser.add_argument('--device', type=str, default="cuda:1", help='Device for Torch model')
        parser.add_argument('--fid_stats', type=str, default="results/celebahq256_fid_stats.npz", help='Path to FID stats npz file (optional)', nargs='?')
        return parser.parse_args()

    if __name__ == "__main__":
        args = get_args()
        torch_inference(
            model_ckpt_path=args.model_ckpt_path,
            out_dir=args.out_dir,
            batch_size=args.batch_size,
            image_size=args.image_size,
            in_channels=args.in_channels,
            denoise_timesteps=args.denoise_timesteps,
            num_generations=args.num_generations,
            cfg_scale=args.cfg_scale,
            num_classes=args.num_classes,
            device=args.device,
            fid_stats=args.fid_stats
        )