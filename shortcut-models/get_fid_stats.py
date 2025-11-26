import numpy as np
import os
from utils.fid import get_fid_network
from utils.datasets import get_dataset

def main():
    # Load FID network
    get_fid_activations = get_fid_network()

    # Load CelebA-HQ256 dataset
    batch_size = 64
    num_images = 10000
    dataset = get_dataset('celebahq256', batch_size, is_train=True)
    
    # Process batches incrementally to avoid OOM
    all_acts = []
    num_batches = (num_images + batch_size - 1) // batch_size
    images_processed = 0
    
    print(f"Processing {num_images} images in {num_batches} batches of {batch_size}...")
    
    for batch_idx in range(num_batches):
        # Get next batch
        batch_images, _ = next(dataset)
        
        # Handle last batch (may be smaller)
        remaining = num_images - images_processed
        if remaining < batch_size:
            batch_images = batch_images[:remaining]
        
        # Apply FID network to this batch
        batch_acts = get_fid_activations(batch_images)
        
        # If output shape is [B, 1, 1, D], squeeze spatial dims
        if batch_acts.ndim == 4 and batch_acts.shape[1:3] == (1, 1):
            batch_acts = batch_acts[..., 0, 0, :] if batch_acts.shape[-1] == 2048 else batch_acts.squeeze((1, 2))
        
        batch_acts = np.array(batch_acts)
        all_acts.append(batch_acts)
        
        images_processed += len(batch_images)
        print(f"Processed batch {batch_idx + 1}/{num_batches} ({images_processed}/{num_images} images)")
    
    # Concatenate all activations
    acts = np.concatenate(all_acts, axis=0)  # shape: (num_images, D)
    print(f"Final activations shape: {acts.shape}")
    
    # Compute mean and covariance
    mu = np.mean(acts, axis=0)
    sigma = np.cov(acts, rowvar=False)

    # Store and save
    fid_stats = {"mu": mu, "sigma": sigma}
    os.makedirs("results", exist_ok=True)
    np.savez("results/celebahq256_fid_stats_10K_samples.npz", mu=mu, sigma=sigma)
    print("Saved FID stats to results/celebahq256_fid_stats_10K_samples.npz")

if __name__ == "__main__":
    main()