import cv2
import sys
import os
from glob import glob
import os.path as osp


def resize_and_crop(img_path, s, out_dir=None):
    """
    Resize the image so the short side is s, then crop the long side to s, and save.
    Args:
        img_path (str): Path to the input image.
        s (int): Target size for both sides.
        out_dir (str, optional): Output directory. If None, saves to original folder.
    """
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not read image: {img_path}")

    h, w = img.shape[:2]
    # Scale so short side is s
    if h < w:
        scale = s / h
        new_h, new_w = s, int(w * scale)
    else:
        scale = s / w
        new_h, new_w = int(h * scale), s
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Crop center to s x s
    start_x = (img_resized.shape[1] - s) // 2
    start_y = (img_resized.shape[0] - s) // 2
    img_cropped = img_resized[start_y:start_y + s, start_x:start_x + s]

    # Prepare output path
    base = os.path.basename(img_path)
    out_folder = out_dir if out_dir is not None else os.path.dirname(img_path)
    os.makedirs(out_folder, exist_ok=True)
    out_path = os.path.join(out_folder, base).replace(".jpg", ".png")
    cv2.imwrite(out_path, img_cropped)
    return out_path

path = sys.argv[1]
size = int(sys.argv[2])
out_dir=None
if len(sys.argv) > 3:
    out_dir = sys.argv[3]
files = []
files.extend(glob(osp.join(path, "*.jpg")))
files.extend(glob(osp.join(path, "*.png")))
for f in files:
    resize_and_crop(f, size, out_dir)