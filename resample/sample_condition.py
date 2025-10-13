from ldm_inverse.condition_methods import get_conditioning_method
from ldm.models.diffusion.ddim import DDIMSampler
from data.dataloader import get_dataset, get_dataloader
from scripts.utils import clear_color, mask_generator
import matplotlib.pyplot as plt
from ldm_inverse.measurements import get_noise, get_operator
from functools import partial
import numpy as np
from model_loader import load_model_from_config, load_yaml
import os
from time import time
import torch
import torchvision.transforms as transforms
import argparse
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from skimage.metrics import peak_signal_noise_ratio as psnr
from metrics import LPIPS, PSNR, SSIM

def get_model(args):
    config = OmegaConf.load(args.ldm_config)
    model = load_model_from_config(config, args.diffusion_config)

    return model

def fmt_mmss(seconds) -> str:
    # ensure non-negative integer seconds
    total = int(max(0, int(seconds)))
    m = total // 60
    s = total % 60
    return f"{m:02d}:{s:02d}"
  
parser = argparse.ArgumentParser()
parser.add_argument('--model_config', type=str)
parser.add_argument('--ldm_config', default="configs/latent-diffusion/ffhq-ldm-vq-4.yaml", type=str)
parser.add_argument('--diffusion_config', default="models/ldm/model.ckpt", type=str)
parser.add_argument('--task_config', default="configs/tasks/gaussian_deblur_config.yaml", type=str)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--save_dir', type=str, required=False)
parser.add_argument('--ddim_steps', default=500, type=int)
parser.add_argument('--ddim_eta', default=0.0, type=float)
parser.add_argument('--n_samples_per_class', default=1, type=int)
parser.add_argument('--ddim_scale', default=1.0, type=float)
parser.add_argument('--sample', default=128, type=int)

args = parser.parse_args()


# Load configurations
task_config = load_yaml(args.task_config)

# Device setting
device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu'
print(f"Device set to {device_str}.")
device = torch.device(device_str)  

# Loading model
model = get_model(args)
sampler = DDIMSampler(model) # Sampling using DDIM

# Prepare Operator and noise
measure_config = task_config['measurement']
operator = get_operator(device=device, **measure_config['operator'])
noiser = get_noise(**measure_config['noise'])
print(f"Operation: {measure_config['operator']['name']} / Noise: {measure_config['noise']['name']}")

# Prepare conditioning method
cond_config = task_config['conditioning']
cond_method = get_conditioning_method(cond_config['method'], model, operator, noiser, **cond_config['params'])
measurement_cond_fn = cond_method.conditioning
print(f"Conditioning sampler : {task_config['conditioning']['main_sampler']}")

# Instantiating sampler
sample_fn = partial(sampler.posterior_sampler, measurement_cond_fn=measurement_cond_fn, operator_fn=operator.forward,
                                        S=args.ddim_steps,
                                        cond_method=task_config['conditioning']['main_sampler'],
                                        conditioning=None,
                                        ddim_use_original_steps=True,
                                        batch_size=args.n_samples_per_class,
                                        shape=[3, 64, 64], # Dimension of latent space
                                        verbose=False,
                                        unconditional_guidance_scale=args.ddim_scale,
                                        unconditional_conditioning=None, 
                                        eta=args.ddim_eta,
                                        only_dps=False)

# Working directory
if args.save_dir:
  out_path = os.path.join(args.save_dir)
  os.makedirs(out_path, exist_ok=True)
  for img_dir in ['input', 'recon', 'progress', 'label']:
      os.makedirs(os.path.join(out_path, img_dir), exist_ok=True)

# Prepare dataloader
data_config = task_config['data']
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))] )
dataset = get_dataset(**data_config, transforms=transform)
dataset.sample(args.sample)
loader = get_dataloader(dataset, batch_size=1, num_workers=0, train=False)

# Exception) In case of inpainting, we need to generate a mask 
if measure_config['operator']['name'] == 'inpainting':
  mask_gen = mask_generator(**measure_config['mask_opt'])

metrics = {"lpips": LPIPS(), 
           "psnr": PSNR(),
           "ssim": SSIM()}

# Do inference
tot_time = 0
results = []
for i, ref_img in enumerate(loader):
    t0 = time()

    print(f"Inference for image {i}")
    fname = str(i).zfill(3)
    results.append([fname])
    ref_img = ref_img.to(device)

    # Exception) In case of inpainting
    if measure_config['operator'] ['name'] == 'inpainting':
      mask = mask_gen(ref_img)
      mask = mask[:, 0, :, :].unsqueeze(dim=0)
      operator_fn = partial(operator.forward, mask=mask)
      measurement_cond_fn = partial(cond_method.conditioning, mask=mask)
      sample_fn = partial(sample_fn, measurement_cond_fn=measurement_cond_fn, operator_fn=operator_fn)

      # Forward measurement model
      y = operator_fn(ref_img)
      y_n = noiser(y)

    else:
      y = operator.forward(ref_img)
      y_n = noiser(y).to(device)

    # Sampling
    samples_ddim, _ = sample_fn(measurement=y_n)
    tot_time = tot_time + time() - t0
    
    x_samples_ddim = model.decode_first_stage(samples_ddim.detach())

    # Post-processing samples
    label = clear_color(y_n)
    reconstructed = clear_color(x_samples_ddim)
    true = clear_color(ref_img)

    if args.save_dir:
      # Saving images
      plt.imsave(os.path.join(out_path, 'input', fname+'_true.png'), true)
      plt.imsave(os.path.join(out_path, 'label', fname+'_label.png'), label)
      plt.imsave(os.path.join(out_path, 'recon', fname+'_recon.png'), reconstructed)

    psnr_cur = psnr(true, reconstructed)
    for met_name, metric in metrics.items():
      score = metric(true, reconstructed)
      results[-1].append(f"{score:.3f}")

for met_name, metric in metrics.items():
  print(f"{met_name}: {metric.mean}")
  
if args.save_dir:
  with open(os.path.join(out_path, "metrics_results.csv"), "w") as f:
    f.write(f"AvgSamplingTime,{fmt_mmss(tot_time/len(dataset))}\n")
    f.write(f"N,{len(dataset)}\n")
    for met_name, metric in metrics.items():
      f.write(f"{met_name},{metric.mean}\n")
    names = [met_name for met_name in metrics]
    f.write(",".join(names)+"\n")
    for res in results:
      f.write(",".join(res)+"\n")

