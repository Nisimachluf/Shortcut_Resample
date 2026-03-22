import os
import os.path as osp
import csv
import argparse
import re
from glob import glob

import numpy as np
from PIL import Image

from metrics import PSNR, SSIM, LPIPS

parser = argparse.ArgumentParser(description="Compare two image collections and compute metrics")
parser.add_argument("--folder1", type=str, required=True, help="First folder (reference images)")
parser.add_argument("--folder2", type=str, required=True, help="Second folder (comparison images)")
parser.add_argument("--out_dir", type=str, required=True, help="Output directory for CSV results")
parser.add_argument("--device", type=str, default="cuda", help="Device for LPIPS computation")

args = parser.parse_args()


def load_image(path):
    """Load image as numpy array in [0, 255] range, shape (H, W, C)."""
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return np.array(img)


def strip_run_suffix(filename):
    """
    Remove _runXXXX suffix from filename if present.
    Examples: 
      image_001_run0001.png -> image_001.png
      image_001.png -> image_001.png
    """
    # Match pattern like _run followed by digits before the extension
    match = re.match(r"(.+?)(_run\d+)?(\.[^.]+)$", filename)
    if match:
        base, _, ext = match.groups()
        return base + ext
    return filename


def get_image_files(folder):
    """Get all image files from folder, sorted by name."""
    patterns = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"]
    files = []
    for pattern in patterns:
        files.extend(glob(osp.join(folder, pattern)))
    return sorted(files)


def match_images(folder1_files, folder2_files):
    """
    Match images from two folders by sorted order.
    
    For folder2, if there are multiple runs (_run0001, _run0002, etc.),
    only keep the first run for each base name, then sort.
    
    Match by position: 1st folder1 image -> 1st folder2 image, etc.
    
    Returns list of tuples: (path1, path2, base_name_from_folder1)
    """
    # folder1: already sorted
    folder1_sorted = folder1_files
    
    # For folder2: filter to first run only, then sort
    folder2_dict = {}
    for path in folder2_files:
        basename = osp.basename(path)
        stripped = strip_run_suffix(basename)
        if stripped not in folder2_dict:
            folder2_dict[stripped] = []
        folder2_dict[stripped].append(path)
    
    # Keep only the first run (sorted by filename) for each base name
    folder2_first_runs = []
    for base_name in sorted(folder2_dict.keys()):
        folder2_first_runs.append(sorted(folder2_dict[base_name])[0])
    
    # Match by position
    n = min(len(folder1_sorted), len(folder2_first_runs))
    pairs = []
    for i in range(n):
        path1 = folder1_sorted[i]
        path2 = folder2_first_runs[i]
        img_name = osp.basename(path1)
        pairs.append((path1, path2, img_name))
    
    if len(folder1_sorted) > n:
        print(f"Warning: {len(folder1_sorted) - n} images from folder1 have no match in folder2")
    if len(folder2_first_runs) > n:
        print(f"Warning: {len(folder2_first_runs) - n} images from folder2 have no match in folder1")
    
    return pairs


def main():
    folder1 = args.folder1
    folder2 = args.folder2
    out_dir = args.out_dir
    device = args.device
    
    if not osp.exists(folder1):
        raise ValueError(f"Folder1 does not exist: {folder1}")
    if not osp.exists(folder2):
        raise ValueError(f"Folder2 does not exist: {folder2}")
    
    os.makedirs(out_dir, exist_ok=True)
    
    # Get image files
    folder1_files = get_image_files(folder1)
    folder2_files = get_image_files(folder2)
    
    print(f"Found {len(folder1_files)} images in folder1")
    print(f"Found {len(folder2_files)} images in folder2")
    
    # Match images
    pairs = match_images(folder1_files, folder2_files)
    print(f"Matched {len(pairs)} image pairs")
    
    if len(pairs) == 0:
        print("No matching pairs found. Exiting.")
        return
    
    # Initialize metrics
    metrics = {
        "lpips": LPIPS(device=device),
        "psnr": PSNR(),
        "ssim": SSIM()
    }
    metric_names = list(metrics.keys())
    
    # Compute metrics for each pair
    details_rows = []
    for path1, path2, img_name in pairs:
        print(f"Processing: {img_name}")
        
        img1 = load_image(path1)
        img2 = load_image(path2)
        
        # Check shapes match
        if img1.shape != img2.shape:
            print(f"Warning: Shape mismatch for {img_name}: {img1.shape} vs {img2.shape}. Skipping.")
            continue
        
        row = {"image": img_name, "time": None}
        for name, metric in metrics.items():
            score = metric(img1, img2)
            row[name] = score
        
        details_rows.append(row)
    
    # Write details.csv
    details_path = osp.join(out_dir, "details.csv")
    with open(details_path, "w", newline="") as f:
        fieldnames = ["image", "time"] + metric_names
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(details_rows)
    
    print(f"Wrote {details_path}")
    
    # Write summary.csv
    n = len(details_rows)
    summary_row = {"num_images": n, "avg_time": None}
    for name in metric_names:
        summary_row[f"avg_{name}"] = sum(r[name] for r in details_rows) / n if n else 0
    
    summary_path = osp.join(out_dir, "summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_row.keys()))
        writer.writeheader()
        writer.writerow(summary_row)
    
    print(f"Wrote {summary_path}")
    print(f"\nSummary:")
    print(f"  Images processed: {n}")
    for name in metric_names:
        print(f"  avg_{name}: {summary_row[f'avg_{name}']:.4f}")


if __name__ == "__main__":
    main()
