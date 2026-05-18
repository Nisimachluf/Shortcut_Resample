import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
sys.path.append("..")

import os.path as osp
import torch
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
import torch.nn as nn

from our_utils import *

def calc_latent_opt(vae, op, z_1: torch.Tensor, y: torch.Tensor, lr, log_grad=False, it=0):
    """
    Optimizes the latent z_1 so that its decoded image x_1, 
    when downsampled by 'op', matches the measurement 'y'.
    """
    it_data = {}
    
    # We run the optimization for several iterations to 'align' the latent
    for i in range(15):
        z_1 = z_1.clone().detach().requires_grad_(True)

        x_1 = vae.decode(z_1)
        Ax_1 = op(x_1)
    
        loss = torch.nn.functional.mse_loss(Ax_1, y, reduction='sum')
        loss.backward()
        
        g = z_1.grad
        # We use a small multiplier (0.02) because VAE gradients can be quite large
        z_1 = z_1 - lr * g * 0.02
        
        if log_grad:
            it_data[i] = z_1.detach().cpu()
            
    # Final Result
    z_data = z_1.detach()
    
    # Decode one last time to get the optimized image x_data
    with torch.no_grad():
        dec_final = vae.decode(z_data)
        x_data = dec_final.sample if hasattr(dec_final, 'sample') else dec_final
    
    # Returning x_data twice to keep the signature compatible with your main loop
    return z_data, x_data, x_data, torch.zeros_like(x_data).cpu().numpy()

# def calc_latent_opt(vae, op, z_1:torch.Tensor, y:torch.Tensor, lr, log_grad=False, it=0):
#     it_data = {}
#     with torch.no_grad():
#         lat_y = vae.encode(y)
#     for i in range(15):
#         z_1 = z_1.clone().detach().requires_grad_(True)
#         # x_1 = vae.decode(z_1)
#         dec = vae.decode(z_1)
#         x_1 = dec.sample if hasattr(dec, 'sample') else dec
#         Ax_1 = op(x_1)
        
#         loss_g_im = None
#         loss = (Ax_1 - y)**2
#         if log_grad:
#             loss_g = (2*(Ax_1-y))[0].abs().detach().cpu().numpy().max(0)
#             loss_g_im = ((loss_g - loss_g.min())/(loss_g.max()-loss_g.min()) * 255).astype("uint8")
#         loss.sum().backward()
#         # print(loss.sum().item())
#         g = z_1.grad
#         # print(z_1.abs().max().item(), g.abs().max().item())
#         z_1 = z_1 - lr*g*0.02
#         it_data[i] = z_1.detach().cpu()
        
#     z_1 = z_1.clone().detach().requires_grad_(True)
#     dist = torch.norm(z_1.flatten(1) - lat_y.flatten(1), dim=1).mean()
#     sim = torch.cosine_similarity(z_1.flatten(1), lat_y.flatten(1), dim=1).mean()
#     dist_loss = (dist - 40.0).abs()
#     sim_loss = (sim - 0.8).abs()
#     cal_loss = dist_loss + sim_loss
#     cal_loss.backward()
#     g = z_1.grad
#     # print(z_1.abs().max().item(), g.abs().max().item())
#     z_1 = z_1 - lr*g
        
        
#     if not os.path.isfile(f"latent_opt_cache_{it}.pt"):
#         torch.save(it_data, f"latent_opt_cache_{it}.pt")
#         if not os.path.isfile(f"latent_y.pt"):
#             torch.save(vae.encode(y).cpu(), f"latent_y.pt")
#     z_data = z_1.detach()
#     x_data = vae.decode(z_data.float())
#     return z_data, x_data, x_1, loss_g_im

def calc_pixel_opt(vae, op, z_1:torch.Tensor, y:torch.Tensor, lr, log_grad=False, inv_opt=None, it=0):
    x_1:torch.Tensor = vae.decode(z_1)
    if inv_opt is not None:
        # if not os.path.isfile(f"bp_opt_cache_{it}.pt"):
            # torch.save({"x_1": x_1.cpu(), "y": y.cpu()}, f"bp_opt_cache_{it}.pt")
        # x_1 = inv_opt(y, x_1)
        Ax = op(x_1) 
        residual = y - Ax
        x_1 = x_1 + inv_opt(residual)
        g_c = torch.zeros_like(x_1).cpu().numpy()
    else:
        for i in range(5):
            x_1 = x_1.clone().detach().requires_grad_(True)
            Ax_1 = op(x_1)
            loss = (Ax_1 - y)**2
            loss.sum().backward()
            g = x_1.grad
            
            g_c = None
            if log_grad:
                g_c = g.clone().detach().cpu().numpy()
                g_c = ((g_c - g_c.min())/(g_c.max()-g_c.min()) * 255).astype("uint8")
                
            x_1 = x_1 - lr*g
    x_data = x_1.detach()
    # z_data = vae.encode(x_data.float())
    enc = vae.encode(x_data.float())
    z_data = (enc.latent_dist.sample() if hasattr(enc, 'latent_dist') else enc)
    return z_data, x_data, x_1, g_c

def shortcut_restoration(vae, model, op, y, z_t=None, ts=0, 
                         dts=16, log_every=0, lr_factor=0.2, 
                         schedule=None, latent_opt_frac=0.0, inv_opt=None):
    if z_t is None:
        z_t = torch.randn_like(vae.encode(y))
        
    if schedule is None:
        schedule = list(zip(*build_step_schedule([1.0], [dts])))
        schedule = [s for s in schedule if s[0] >= ts]

    intermediates = {"x_t": [], "x_1": [], "x_data": [], "grads": [],}
    for i, (ti, dts) in tqdm(enumerate(schedule), total=len(schedule)):
        lr = (1-ti/dts)**lr_factor
        
        t = calc_t(ti, dts)
        v = calc_v(model, z_t, t, dts)
        z_1 = z_t + (1-t)*v
        print(z_1.shape, y.shape)
        
        if t > latent_opt_frac:
            z_data, x_data, x_1, g_c = calc_latent_opt(vae, op, z_1, y, lr, log_grad=log_every and i % log_every == 0, it=i)
        else:
            z_data, x_data, x_1, g_c = calc_pixel_opt(vae, op, z_1, y, lr, log_grad=log_every and i % log_every == 0, inv_opt=inv_opt, it=i)
        
        if log_every and i % log_every == 0:
            intermediates["x_data"].append(to_numpy(x_data.detach().cpu()))
            intermediates["x_1"].append(to_numpy(x_1.detach().cpu()))
            intermediates["grads"].append(g_c)
        
        if i < len(schedule)-1:
            z_t = stochastic_encoding(z_data, *schedule[i+1])
        else:
            z_t = z_data
        if log_every and i % log_every == 0:
                intermediates["x_t"].append(to_numpy(decode(z_t,vae, False)))
    return z_t, intermediates
    
def shortcut_refinement(refinement_rounds, vae, model, op, y, z_t=None, lr_factor=0.2):
    dts = 2
    ts = 0
    for round in range(refinement_rounds):
        z_t, intermediates = shortcut_restoration(vae, model, op, y, z_t=z_t, ts=ts, 
                             dts=dts, log_every=0, lr_factor=lr_factor, latent_opt_frac=0.85,
                             schedule=None)
        dts *= 2
        ts = dts/4
        # print(f"Refinement round {round+1}/{refinement_rounds} done. Next round will start at t={ts} with dt={dts}.")
        if round < refinement_rounds-1:
            z_t = stochastic_encoding(z_t, calc_t(ts, dts), dts)
    return z_t, intermediates