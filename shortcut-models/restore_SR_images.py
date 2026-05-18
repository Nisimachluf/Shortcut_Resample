import os
import os.path as osp
import csv
import json
import time
import argparse
from glob import glob
from random import shuffle, seed

import numpy as np
from PIL import Image

from our_utils import *
from metrics import PSNR, SSIM, LPIPS
from our_restoration_method import shortcut_restoration, build_step_schedule, shortcut_refinement
from down_sampler import Resizer, SuperResolutionOperator
seed(1210)

parser = argparse.ArgumentParser(description="Shortcut image restoration")
parser.add_argument("--run_name",   type=str,   default="run",           help="Base name for the output run folder")
parser.add_argument("--out_dir",    type=str,   default="shortcut_restoration_results_SR", help="Root output directory")
parser.add_argument("--images_dir", type=str,   default="/private/tirer-lab/coheny78/CelebA/validation", help="Directory containing input images")
parser.add_argument("--num_images", type=int,   default=100,             help="Number of images to process")
parser.add_argument("--ts",         type=int,   default=0,               help="Start timestep")
parser.add_argument("--dts",        type=int,   default=64,              help="Number of diffusion steps")
parser.add_argument("--log_every",  type=int,   default=8,               help="Log intermediates every N steps (0 = disabled)")
parser.add_argument("--lr_factor",  type=float, default=0.2,             help="Exponent for the per-step learning rate schedule: lr=(1-t)^lr_factor")
parser.add_argument("--latent_opt_phase", type=float, default=0.85,      help="Fraction of steps to use for latent optimization phase")
parser.add_argument("--device",     type=str,   default="cuda",          help="Torch device")
parser.add_argument("--log_images", action="store_true", default=False,  help="Save input/distorted/restored/intermediate images")
parser.add_argument("--refine", action="store_true", default=False,  help="Enable refinement rounds")

args = parser.parse_args()

if not args.log_images:
    args.log_every = 0

def get_result_dir(out_dir, run_name):
    """Return a unique run directory under out_dir, adding a numeric suffix if needed."""
    candidate = osp.join(out_dir, run_name)
    suffix = 1
    while osp.exists(candidate):
        candidate = osp.join(out_dir, f"{run_name}_{suffix}")
        suffix += 1
    subdirs = {
        "inputs":        osp.join(candidate, "inputs"),
        "distorted":     osp.join(candidate, "distorted"),
        "restored":      osp.join(candidate, "restored"),
        "intermediates": osp.join(candidate, "intermediates"),
    }
    for d in subdirs.values():
        os.makedirs(d, exist_ok=True)
    print(f"Run directory: {candidate}")
    return candidate, subdirs


run_name = args.run_name
out_dir  = args.out_dir
run_dir, dirs = get_result_dir(out_dir, run_name)

# Save input arguments to JSON
args_path = osp.join(run_dir, "args.json")
with open(args_path, "w") as f:
    json.dump(vars(args), f, indent=2)
print(f"Arguments saved to {args_path}")


def save_img(arr, path):
    """Save a numpy array (HWC or CHW, float or uint8) as JPEG."""
    arr = np.squeeze(arr)
    if arr.ndim == 3 and arr.shape[0] in (1, 3):   # CHW -> HWC
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        if arr.max() <= 1.0:
            arr = arr * 255
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    img = arr.squeeze()
    if img.ndim == 2:
        Image.fromarray(img, mode="L").save(path)
    else:
        Image.fromarray(img).save(path)


def save_tensor_img(t, path):
    """Save a torch tensor image (BCHW or CHW, float [-1,1]) as JPEG."""
    arr = to_numpy(t)
    save_img(arr, path)

metrics = {"lpips": LPIPS(), 
           "psnr": PSNR(),
           "ssim": SSIM()}

device = args.device
images_dir = args.images_dir
images_paths = glob(osp.join(images_dir, "*.png"))
shuffle(images_paths)

kernel, noiser = get_debluring_tools()
image_SHAPE = (1, 3, 256, 256)
scale_factor = 4
sr_op = SuperResolutionOperator(in_shape=image_SHAPE, scale_factor=scale_factor, device=device)

model, vae = load_models()

details_path = osp.join(run_dir, "details.csv")
summary_path = osp.join(run_dir, "summary.csv")

details_rows = []
metric_names = list(metrics.keys())
inv_opt_func = lambda residual: sr_op.transpose(residual)
# inv_opt_func = lambda y, x: sr_op.project(data=x, measurement=y)
for imgp in images_paths[:args.num_images]:
    img_name = osp.splitext(osp.basename(imgp))[0]
    img_t, img = read_image_to_tensor(imgp, return_np=True)
    img_t = img_t.to(device)
    y = noiser(sr_op.forward(img_t))
    print(f"Going from {img_t.shape} to {y.shape}")


    # save input (clean) and distorted images
    if args.log_images:
        save_img(img, osp.join(dirs["inputs"], f"{img_name}.jpg"))
        save_tensor_img(y, osp.join(dirs["distorted"], f"{img_name}.jpg"))

    t_start = time.time()
    # schedule = list(zip(*build_regularly_decaying_schedule(args.dts)))
    if args.refine:
        restored_img_z, intermediates = shortcut_refinement(int(np.log2(args.dts)), vae, model, sr_op.forward, y, lr_factor=args.lr_factor) #, schedule=schedule)
    else:
        # inv_opt = lambda y, x: sr_op.project(data=x, measurement=y)
        target_latent_shape = (1, 4, 32, 32) 
        z_t = torch.randn(target_latent_shape).to(device)
        restored_img_z, intermediates = shortcut_restoration(vae, model, sr_op.forward, y,z_t=z_t, ts=args.ts, dts=args.dts, log_every=args.log_every, lr_factor=args.lr_factor, latent_opt_frac=args.latent_opt_phase, inv_opt=inv_opt_func) #, schedule=schedule,)
    
    elapsed = time.time() - t_start

    restored_image = decode(restored_img_z, vae, rescale=True)
    test_img = restored_image[0] # Shape: (C, H, W)
    print(f"Final Metric Input Shape: {test_img.shape}")


    # # save restored image
    if args.log_images:
        save_img(test_img if isinstance(test_img, (list, np.ndarray)) and np.array(test_img).ndim > 3
                 else test_img, osp.join(dirs["restored"], f"{img_name}.jpg"))

    # save intermediates (last logged frame per key when log_every > 0)
    if args.log_images:
        for key, frames in intermediates.items():
            if frames:
                for i, frame in enumerate(frames):
                    save_img(frame, osp.join(dirs["intermediates"], f"{img_name}_{key}_{i}.jpg"))

    row = {"image": osp.basename(imgp), "time": elapsed}
    for name, metric in metrics.items():
        score = metric(img, restored_image[0])
        row[name] = score
        # print(name, score)
    details_rows.append(row)

# Write details.csv
with open(details_path, "w", newline="") as f:
    fieldnames = ["image", "time"] + metric_names
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(details_rows)

# Write summary.csv
n = len(details_rows)
avg_time = sum(r["time"] for r in details_rows) / n if n else 0
summary_row = {"num_images": n, "avg_time": avg_time}
for name in metric_names:
    summary_row[f"avg_{name}"] = sum(r[name] for r in details_rows) / n if n else 0

with open(summary_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(summary_row.keys()))
    writer.writeheader()
    writer.writerow(summary_row)

print(f"Results saved to {run_dir}")
