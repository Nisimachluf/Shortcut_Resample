from datasets import load_dataset
import random
import os
from PIL import Image
import io
import os.path as osp
from tqdm import tqdm
# ==== CONFIG ====
SEED = 42
NUM_SAMPLES = 100
SAVE_DIR = osp.join("datasets", "ImageNet")
os.makedirs(SAVE_DIR, exist_ok=True)
# Load streamed validation split
dataset = load_dataset("Leonardo6/imagenet256-sdvae", split="train", streaming=True,)
for i, item in tqdm(enumerate(dataset.shuffle(seed=42)), total=NUM_SAMPLES, desc="Downloadig ImageNet"):
    if i == NUM_SAMPLES:
        break
    img = item["image.png"]  # already a PIL.Image
    img.save(os.path.join(SAVE_DIR, f"imagenet_{i:04d}.jpg"))

print("✅ Done! Saved", NUM_SAMPLES, "images to", SAVE_DIR)
