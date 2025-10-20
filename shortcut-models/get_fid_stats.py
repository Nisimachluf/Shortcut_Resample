import numpy as np
import os
from utils.fid import get_fid_network
from utils.datasets import get_dataset

def main():
    # Load FID network
    get_fid_activations = get_fid_network()

    # Load CelebA-HQ256 dataset
    batch_size = 64
    num_images = 512
    dataset = get_dataset('celebahq256', batch_size, is_train=False)
    images = []

    # Collect images
    while len(images) * batch_size < num_images:
        batch_images, _ = next(dataset)
        images.append(batch_images)
    images = np.concatenate(images, axis=0)[:num_images]  # shape: (512, H, W, C)
    print(images.shape)
    # Apply FID network
    acts = get_fid_activations(images)
    print(acts.shape)
    # If output shape is [B, 1, 1, D], squeeze spatial dims
    if acts.ndim == 4 and acts.shape[1:3] == (1, 1):
        acts = acts[..., 0, 0, :] if acts.shape[-1] == 2048 else acts.squeeze((1,2))
    acts = np.array(acts)  # shape: (512, D)

    # Compute mean and covariance
    mu = np.mean(acts, axis=0)
    sigma = np.cov(acts, rowvar=False)

    # Store and save
    fid_stats = {"mu": mu, "sigma": sigma}
    os.makedirs("results", exist_ok=True)
    np.savez("results/celebahq256_fid_stats.npz", mu=mu, sigma=sigma)
    print("Saved FID stats to results/celebahq256_fid_stats.npz")

if __name__ == "__main__":
    main()