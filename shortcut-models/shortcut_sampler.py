"""SAMPLING ONLY."""

import torch
import torch.fft as fft
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from functools import partial
from scripts.utils import *
from helper_bp import A, A_adjoint, gaussian_kernel
from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like, \
    extract_into_tensor
from skimage.metrics import peak_signal_noise_ratio as psnr
from scripts.utils import clear_color
from sampling_tools import denoising_step, build_step_schedule, shortest_plan_to_end

from typing import Set

def action_indices_on_coarse_grid(N: int, base: int = 128, interval: int = 10) -> Set[int]:
    """
    Return set of coarse indices j in [0, N-1] that are the nearest mappings
    of original action positions {0, interval, 2*interval, ...} on a timeline of length `base`.
    Rounding rule: j = int(a * N / base + 0.5). Clamped to [0, N-1].
    """
    if N <= 0 or (N & (N - 1)) != 0:
        raise ValueError("N must be a positive power of two (e.g. 1,2,4,8,16,32,64...).")
    indices = set()
    for a in range(0, base, interval):
        j = int((a * N) / base + 0.5)
        if j >= N:
            j = N - 1
        indices.add(j)
    return indices

def is_closest_action(N: int, i: int, base: int = 128, interval: int = 10) -> bool:
    """
    Return True if coarse index i (0 <= i < N) is the nearest coarse position
    to at least one original action (multiples of `interval` on [0..base-1]).
    """
    if not (0 <= i < N):
        raise ValueError("i must be in range(0, N)")
    return i in action_indices_on_coarse_grid(N, base=base, interval=interval)

class ShortcutSampler(object):
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        # self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule

    # def register_buffer(self, name, attr):
    #     if type(attr) == torch.Tensor:
    #         if attr.device != torch.device("cuda"):
    #             attr = attr.to(torch.device("cuda"))
    #     setattr(self, name, attr)

    # def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
    #     if verbose: print("Generating schedule:")
    #     self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
    #                                               num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=verbose)
    #     alphas_cumprod = self.model.alphas_cumprod
    #     assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
    #     to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

    #     self.register_buffer('betas', to_torch(self.model.betas))
    #     self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
    #     self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))

    #     # calculations for diffusion q(x_t | x_{t-1}) and others
    #     self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
    #     self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
    #     self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
    #     self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
    #     self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

    #     # ddim sampling parameters
    #     if ddim_num_steps < 1000:
    #       ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
    #                                                                                 ddim_timesteps=self.ddim_timesteps,
    #                                                                                 eta=ddim_eta,verbose=verbose)
    #       self.register_buffer('ddim_sigmas', ddim_sigmas)
    #       self.register_buffer('ddim_alphas', ddim_alphas)
    #       self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
    #       self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
    #     sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
    #           (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
    #                       1 - self.alphas_cumprod / self.alphas_cumprod_prev))
    #     self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)


    def sample(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs
               ):
        """
        Sampling wrapper function for UNCONDITIONAL sampling.
        """

        # if conditioning is not None:
        #     if isinstance(conditioning, dict):
        #         cbs = conditioning[list(conditioning.keys())[0]].shape[0]
        #         if cbs != batch_size:
        #             print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
        #     else:
        #         if conditioning.shape[0] != batch_size:
        #             print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        raise NotImplementedError("Need to implement shortcut sampling here")
        # self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')

        samples, intermediates = self.ddim_sampling(conditioning, size,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    mask=mask, x0=x0,
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=x_T,
                                                    log_every_t=log_every_t,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    )
        return samples, intermediates


    def posterior_sampler(self, measurement, measurement_cond_fn, operator_fn,
               batch_size,
               shape,
               cond_method=None,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               only_dps=False,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs
               ):
        """
        Sampling wrapper function for inverse problem solving.
        """
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        # self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        size = (batch_size,C, H, W)
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')

        if cond_method is None or cond_method == 'resample':
            samples, intermediates = self.resample_sampling(measurement, measurement_cond_fn,
                                                    conditioning, size,
                                                        operator_fn=operator_fn,
                                                        callback=callback,
                                                        img_callback=img_callback,
                                                        quantize_denoised=quantize_x0,
                                                        mask=mask, x0=x0,
                                                        noise_dropout=noise_dropout,
                                                        temperature=temperature,
                                                        score_corrector=score_corrector,
                                                        corrector_kwargs=corrector_kwargs,
                                                        x_T=x_T,
                                                        log_every_t=log_every_t,
                                                        unconditional_guidance_scale=unconditional_guidance_scale,
                                                        unconditional_conditioning=unconditional_conditioning,
                                                        verbose=verbose,
                                                        only_dps=only_dps
                                                        )
            
        else:
            raise ValueError(f"Condition method string '{cond_method}' not recognized.")
        
        return samples, intermediates

    @staticmethod
    def calc_a_t(step, step_size):
        return step / step_size
    
    @staticmethod
    def calc_a_prev(step, step_size):
        return ShortcutSampler.calc_a_t(step-1, step_size)
    
    def resample_sampling(self, measurement, measurement_cond_fn, cond, shape, operator_fn=None,
                     inter_timesteps=10, x_T=None,
                     callback=None, timesteps=None, quantize_denoised=False,
                     mask=None, x0=None, img_callback=None, log_every_t=100,
                     temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                     unconditional_guidance_scale=1., unconditional_conditioning=None,verbose=False,only_dps=False):
        """
        DDIM-based sampling function for ReSample.

        Arguments:
            measurement:            Measurement vector y in y=Ax+n.
            measurement_cond_fn:    Function to perform DPS. 
            operator_fn:            Operator to perform forward operation A(.)
            inter_timesteps:        Number of timesteps to perform time travelling.

        """

        inter_timesteps = 0
        gamma = 1
        device = self.model.device
        b = shape[0]
        kernel = gaussian_kernel()
        if x_T is None:
            noise = torch.randn(shape, device=device)
            img = self.generate_x0_warm(measurement, noise, kernel)
            # img=noise
        else:
            img = x_T
        
        img = img.requires_grad_() # Require grad for data consistency

        if timesteps is None:
            timesteps = 32 # self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
            if verbose: print("got None timesteps {}".format(f"using default timesteps {timesteps}"))

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        # flip the timesteps to count backward
        time_range = reversed(range(0,timesteps))
        total_steps = timesteps
        # Need for measurement consistency
        # alphas = self.model.alphas_cumprod if ddim_use_original_steps else self.ddim_alphas 
        # alphas_prev = self.model.alphas_cumprod_prev if ddim_use_original_steps else self.ddim_alphas_prev
        # betas = self.model.betas
        
        # schedule = list(zip(*build_step_schedule((32/128,16/128, 80/128, ), (4, 8, 64))))
        schedule = list(zip(*build_step_schedule((1, ), (timesteps,))))
        iterator = tqdm(schedule, desc='Shortcut Sampler', total=len(schedule))

        # each of the time steps starting from the original ddpm number of steps, ddpm steps size
        # x_t = (1-t)x_0 + 
        kernel = gaussian_kernel()

        for i, (step, dts) in enumerate(iterator):
            # a_t is the fractional path in [0, 1] from pure noise(0) to real image (1)
            t = ShortcutSampler.calc_a_t(step, dts)
            t_prev = ShortcutSampler.calc_a_t(step, dts) 
            a_t = torch.full((b, 1, 1, 1), t, device=device, requires_grad=False) # Needed for ReSampling
            a_prev = torch.full((b, 1, 1, 1), t_prev, device=device, requires_grad=False) # Needed for ReSampling
            
            
            # i counts from 0 to 499 (the number of ddim steps)        
            # Instantiating parameters
            # index = total_steps - i - 1 #index the actual ddim steps, backwards
            # ts = torch.full((b,), step, device=device, dtype=torch.long) # batch size of current time step
            # a_t = torch.full((b, 1, 1, 1), alphas[index], device=device, requires_grad=False) # Needed for ReSampling
            # a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device, requires_grad=False) # Needed for ReSampling
            # b_t = torch.full((b, 1, 1, 1), betas[index], device=device, requires_grad=False)            

            if mask is not None:
                if verbose: print("Masking img")
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
                img = img_orig * mask + (1. - mask) * img
            else:
                if verbose: print("Not using mask")
            
            # Unconditional sampling step
            # pred_x0 is from DDIM, pseudo_x0 is computing \hat{x}_0 using Tweedie's formula
            # returns:
            # x_prev - the prediction of x_{t-1}
            # pred_x0, - estimation of x0 using (x_t-\sqrt(1-at)e(x_t, t))/a_t
            # pseudo_x0 - estimation of x0 using (x_t-(1-at)e(x_t, t))/a_t
            if verbose: print(f"Runnign shortcut sampling with {step}/{dts}")
            out, pred_x0, pseudo_x0 = self.p_sample_shortcut(img, step, dts, verbose=verbose)
            
            # Conditioning step
            lr = step / dts*0.5  # alpha_t * 0.5 in the original reample implementation
            if verbose: print(f"Running DPS conditioning with LR={lr:.4f}")
            img, _ = measurement_cond_fn(
                                            x_prev=img, # x_prev is x_t
                                            x_t=out, # x_t is x_{t-1}
                                            x_0_hat=pseudo_x0,
                                            measurement=measurement, # thats the noise sample
                                            scale=lr, # For DPS learning rate / scale
                                            noisy_measurement=measurement, #not used in DPS contioning
                                            verbose=verbose
                                            )
            if only_dps:
                continue
            # Instantiating time-travel parameters
            splits = 3 # TODO: make this not hard-coded
            phase_1 = 1/splits
            phase_2 = 2/splits

            # Performing time-travel if in selected indices
            if a_t >= phase_1:   
                x_t = img.detach().clone()

                # Performing only every 10 steps (or so)
                # TODO: also make this not hard-coded
                if i % 10 == 0 :
                # if is_closest_action(dts,i): # % 1 == 0 :
                    if verbose: print(f"Index = {i}")
                    if verbose: print(f"Iterating over  = {list(range(i, min(i+inter_timesteps, -1)))}")
                    for k in range(i, min(i+inter_timesteps, timesteps-1)):
                        step_ = list( reversed(timesteps))[k+1]
                        ts_ = torch.full((b,), step_, device=device, dtype=torch.long)
                        index_ = total_steps - k - 1
                        if verbose: print(f"Trio: step_={step_.item()}, ts_={ts_.item()}, index_={index_}")
                        # Obtain x_{t-k}
                        if verbose: print(f"Runnign DDIM sampling with t={ts_.item()}, index={index_}")
                        img, pred_x0, pseudo_x0 = self.p_sample_ddim(img, cond, ts_, index=index_, use_original_steps=ddim_use_original_steps,
                                            quantize_denoised=quantize_denoised, temperature=temperature,
                                            noise_dropout=noise_dropout, score_corrector=score_corrector,
                                            corrector_kwargs=corrector_kwargs,
                                            unconditional_guidance_scale=unconditional_guidance_scale,
                                            unconditional_conditioning=unconditional_conditioning,
                                            verbose=verbose)
                        
                    # Some arbitrary scheduling for sigma
                    if dts-step > 1: #not last step
                        sigma = gamma*(1 - a_prev) / (1 - a_t) * (1 - a_t / a_prev)  
                    else:
                        sigma = 0.5

                    # Pixel-based optimization for second stage
                    if a_t <= phase_2: 
                        
                        # Enforcing consistency via pixel-based optimization
                        pseudo_x0 = pseudo_x0.detach() 
                        pseudo_x0_pixel = self.model.decode_first_stage(pseudo_x0) # Get \hat{x}_0 into pixel space

                        # opt_var = self.pixel_optimization(measurement=measurement, 
                        #                                   x_prime=pseudo_x0_pixel,
                        #                                   operator_fn=operator_fn,
                        #                                      verbose=verbose)

                        # opt_var = self.iterative_bp_with_reg(measurement=measurement, 
                        #                                   x_prime=pseudo_x0_pixel)

                        opt_var = self.solve_bp_fft_operator(y=measurement, x0=pseudo_x0_pixel,kernel=kernel)
                        
                        opt_var = self.model.encode_first_stage(opt_var) # Going back into latent space

                        img = self.stochastic_resample(pseudo_x0=opt_var, x_t=x_t, a_t=a_prev, sigma=sigma)
                        # img = self.flow_matching_noise_injection(x0_y=opt_var,t_n=a_prev)
                        img = img.requires_grad_() # Seems to need to require grad here

                    # Latent-based optimization for third stage
                    elif a_t > phase_2: # Needs to (possibly) be tuned

                        # Enforcing consistency via latent space optimization
                        pseudo_x0, _ = self.latent_optimization(measurement=measurement,
                                                             z_init=pseudo_x0.detach(),
                                                             operator_fn=operator_fn,
                                                             verbose=verbose)


                        sigma = gamma * (1-a_prev)/(1 - a_t) * (1 - a_t / a_prev) # Change the 40 value for each task

                        img = self.stochastic_resample(pseudo_x0=pseudo_x0, x_t=x_t, a_t=a_prev, sigma=sigma)
                        # img = self.flow_matching_noise_injection(x0_y=opt_var,t_n=a_prev)

            # Callback functions if needed
            if callback: callback(i)
            if img_callback: img_callback(pred_x0, i)
            if i % log_every_t == 0 or i == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)       
        
        if not only_dps:
            psuedo_x0, _ = self.latent_optimization(measurement=measurement,
                                                             z_init=img.detach(),
                                                             operator_fn=operator_fn,
                                                             verbose=verbose)
            img = psuedo_x0.detach().clone()
            
        return img, intermediates


    def pixel_optimization(self, measurement, x_prime, operator_fn, eps=1e-3, max_iters=2000, verbose=False):
        """
        Function to compute argmin_x ||y - A(x)||_2^2

        Arguments:
            measurement:           Measurement vector y in y=Ax+n.
            x_prime:               Estimation of \hat{x}_0 using Tweedie's formula
            operator_fn:           Operator to perform forward operation A(.)
            eps:                   Tolerance error
            max_iters:             Maximum number of GD iterations
        """

        loss = torch.nn.MSELoss() # MSE loss

        opt_var = x_prime.detach().clone()
        opt_var = opt_var.requires_grad_()
        optimizer = torch.optim.AdamW([opt_var], lr=1e-2) # Initializing optimizer
        measurement = measurement.detach() # Need to detach for weird PyTorch reasons

        # Training loop
        stopped = False
        for _ in range(max_iters):
            optimizer.zero_grad()
            
            measurement_loss = loss(measurement, operator_fn( opt_var ) ) 
            
            measurement_loss.backward() # Take GD step
            optimizer.step()

            # Convergence criteria
            if measurement_loss < eps**2: # needs tuning according to noise level for early stopping
                if verbose: print("Pixel optimization reached early stopping")
                stopped = True
                break
        if verbose and not stopped:
            print(f"Pixel optimization reached max iterations ({max_iters}), last loss {measurement_loss:.3e} (eps={(eps**2):.3e})")
        return opt_var
    
    
    # def iterative_bp_with_reg(self, measurement, x_prime, lam=1.0, alpha=0.1, n_steps=4):
    #     """
    #     Single-step back-projection with optional L2 regularization toward x_prime.
        
    #     Args:
    #         measurement: [B, C, H, W]  - observed measurement (blurred + noisy)
    #         x_prime:     [B, C, H, W]  - current estimate (e.g. from Shortcut)
    #         A:           function       - forward operator (e.g., blur)
    #         A_adjoint:   function       - adjoint of A
    #         lam:         float          - back-projection step size
    #         alpha:       float          - L2 regularization strength toward x_prime
    #     """
    # # Computsidual = measurement - blur_operator(x_prime)
    #     # correction = blur_adjoint(residual)
    #     # x = x_prime + lam * correction
        
    #     residual = measurement - blur_operator(x) #
    #     A_adjoint = blur_adjoint(A)
    #     reg_term_1 = x - x_prime #L2 regularization gradient
    #     # reg_term_2 = laplacian(x)
    #     # x = x + lam * correction - alpha * (0.2 * reg_term_1 + 0.8 * reg_term_2)
    #     # x = x + lam * correction - alpha * reg_term_1
    #     x = A
    #     return x

    def generate_x0_warm(self, y, x0, kernel, sigma=0.9, d=0.7, device='cuda'):
        """
        Generate a warm-start x0|y for Shortcut latent flow with a weighted 'big jump' start.

        Args:
            y: torch.Tensor, the measurement or input image
            x0: initial guess (optional, can be zeros)
            kernel: your forward operator kernel
            sigma: float, noise level to add on top of x_hat
            d: float in [0,1], mixing factor between pure noise and x0|y
            device: str, device to put the tensor on

        Returns:
            x0_warm: torch.Tensor, warm initialization for the sampler
        """
        with torch.no_grad():
            # Solve back-projection to get x0|y
            x0_pixel_prior = self.model.differentiable_decode_first_stage(x0)
            x_hat_pixel = self.solve_bp_fft_operator(y, x0_pixel_prior, kernel)  # [B, C, H, W]
            x_hat_latent = self.model.encode_first_stage(x_hat_pixel) 
            x_hat_latent = x_hat_latent.to(device)

            # Generate random Gaussian noise
            noise_latent = torch.randn_like(x_hat_latent) * sigma

            # Weighted combination: "big jump" start
            x0_warm = (1 - d) * x_hat_latent + d * noise_latent

        return x0_warm


    def solve_bp_fft_operator(self, y, x0, kernel, rho=1.0, eps=1e-12):
        """ 
        let k be the kernel
        A^T be the adjoint of A, K^* be the adjoint in frequency domain
        for rho >0, bp closed form is : x = (A^T A + rho I)^-1 (A^T y+ rho  x0)
        A is convolution operator with kernel k, we get:

        In pixel space:
        A^T A x = (x * k) * k^T
        A^Ty = y *k^T                                         # convolution with flipped kernel

        In frequency domain: convolution becomes multiplication:
        F{Ay} = F{k^T * y} = K^* ⋅ Y
        F{A^TAx} = F{(X*K) * k^T} = (X*k) * K^* = X * |K|^2
        so our solution is:
        X =K^* ⋅ Y + rho X0 / (|K|^2 + rho)
        """

        B, C, H, W = y.shape
        # pad kernel to image size
        kh, kw = kernel.shape[-2:]
        kernel_padded = F.pad(kernel, (0, W - kw, 0, H - kh))
        if kernel_padded.shape[0] == 1 and C > 1:
            kernel_padded = kernel_padded.repeat(C,1,1,1)

        if not hasattr(self, 'k_f_cache') or self.k_f_cache.shape[-2:] != (H, W//2 + 1):
            k_f = fft.rfft2(kernel_padded.reshape(C, H, W))
            self.k_f_cache = k_f
        else:
            k_f = self.k_f_cache
        y_f = fft.rfft2(y.reshape(B*C,H,W))
        x0_f = fft.rfft2(x0.reshape(B*C,H,W))
        k_f = k_f.repeat(B, 1, 1).reshape(B*C, H, W//2 + 1)

        numerator = torch.conj(k_f)*y_f + rho*x0_f
        denominator = (k_f.abs()**2) + rho  + eps
        x_f = numerator / denominator
        # x = fft.ifft2(x_f).real
        x = fft.irfft2(x_f, s=(H, W))
        return x.reshape(B,C,H,W)



    def latent_optimization(self, measurement, z_init, operator_fn, eps=1e-3, max_iters=500, lr=None, verbose=False):

        """
        Function to compute argmin_z ||y - A( D(z) )||_2^2

        Arguments:
            measurement:           Measurement vector y in y=Ax+n.
            z_init:                Starting point for optimization
            operator_fn:           Operator to perform forward operation A(.)
            eps:                   Tolerance error
            max_iters:             Maximum number of GD iterations
        
        Optimal parameters seem to be at around 500 steps, 200 steps for inpainting.

        """

        # Base case
        if not z_init.requires_grad:
            z_init = z_init.requires_grad_()

        if lr is None:
            lr_val = 5e-3
        else:
            lr_val = lr.item()

        loss = torch.nn.MSELoss() # MSE loss
        optimizer = torch.optim.AdamW([z_init], lr=lr_val) # Initializing optimizer ###change the learning rate
        measurement = measurement.detach() # Need to detach for weird PyTorch reasons

        # Training loop
        init_loss = 0
        losses = []
        stopped = False
        for itr in range(max_iters):
            optimizer.zero_grad()
            output = loss(measurement, operator_fn( self.model.differentiable_decode_first_stage( z_init ) ))          

            if itr == 0:
                init_loss = output.detach().clone()
                
            output.backward() # Take GD step
            optimizer.step()
            cur_loss = output.detach().cpu().numpy() 
            
            # Convergence criteria

            if itr < 200: # may need tuning for early stopping
                losses.append(cur_loss)
            else:
                losses.append(cur_loss)
                if losses[0] < cur_loss:
                    if verbose: print("Latent optimization reached early stopping (flat loss)")
                    stopped = True
                    break
                else:
                    losses.pop(0)
                    
            if cur_loss < eps**2:  # needs tuning according to noise level for early stopping
                if verbose: print("Latent optimization reached early stopping (under treshold)")
                stopped = True
                break

        if verbose and not stopped:
            print(f"Latent optimization reached max iterations ({max_iters}), last loss {cur_loss:.3e} (eps={(eps**2):.3e})")
        return z_init, init_loss       


    def stochastic_resample(self, pseudo_x0, x_t, a_t, sigma):
        """
        Function to resample x_t based on ReSample paper.
        """
        device = self.model.device
        noise = torch.randn_like(pseudo_x0, device=device)
        return (sigma * a_t.sqrt() * pseudo_x0 + (1 - a_t) * x_t)/(sigma + 1 - a_t) + noise * torch.sqrt(1/(1/sigma + 1/(1-a_t)))


    def ddim_sampling(self, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,):
        """
        Function for unconditional sampling using DDIM.
        """

        device = self.model.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            if mask is not None:
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
                img = img_orig * mask + (1. - mask) * img

            outs = self.p_sample_ddim(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                      quantize_denoised=quantize_denoised, temperature=temperature,
                                      noise_dropout=noise_dropout, score_corrector=score_corrector,
                                      corrector_kwargs=corrector_kwargs,
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=unconditional_conditioning)
            img, pred_x0 = outs
            if callback: callback(i)
            if img_callback: img_callback(pred_x0, i)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)

        return img, intermediates


    def p_sample_ddim(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None, verbose=False):
        if verbose: print("Running p_sample_ddim")
        b, *_, device = *x.shape, x.device

        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            if verbose: print(f"Using unconditional ddim sampling (x={x.shape}, t={t.item()}, c={c})")
            e_t = self.model.apply_model(x, t, c)
        else:
            if verbose: print(f"Unconditional Conditioning: {unconditional_conditioning}, unconditional_guidance_scale: {unconditional_guidance_scale}")
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            c_in = torch.cat([unconditional_conditioning, c])
            e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

        if score_corrector is not None:
            assert self.model.parameterization == "eps"
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas

        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)
        if verbose: print(f"a_t={a_t.item():.3f}, a_pred={a_prev.item():.3f}, sigma_t={sigma_t.item():.3f}")
        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise

        # Computing \hat{x}_0 via Tweedie's formula
        pseudo_x0 = (x - sqrt_one_minus_at**2 * e_t) / a_t.sqrt()
        return x_prev, pred_x0, pseudo_x0
    
    def p_sample_shortcut(self, x, t, denoising_timesteps, verbose=False):
        num_classes = 1
        batch_size = x.shape[0]
        beta=0.9
        labels = torch.randint(0, num_classes, (batch_size,), device=self.model.device)
        x_prev, pseudo_x0 = denoising_step(self.model, x, t, denoising_timesteps, labels, 0, 1, self.model.device)
        
        return x_prev, x_prev, pseudo_x0
        


    def stochastic_encode(self, x0, t, use_original_steps=False, noise=None):
        # fast, but does not allow for exact reconstruction
        # t serves as an index to gather the correct alphas
        if use_original_steps:
            sqrt_alphas_cumprod = self.sqrt_alphas_cumprod
            sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod
        else:
            sqrt_alphas_cumprod = torch.sqrt(self.ddim_alphas)
            sqrt_one_minus_alphas_cumprod = self.ddim_sqrt_one_minus_alphas

        if noise is None:
            noise = torch.randn_like(x0)
        return (extract_into_tensor(sqrt_alphas_cumprod, t, x0.shape) * x0 +
                extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise)


    def decode(self, x_latent, cond, t_start, unconditional_guidance_scale=1.0, unconditional_conditioning=None,
               use_original_steps=False):

        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        timesteps = timesteps[:t_start]

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='Decoding image', total=total_steps)
        x_dec = x_latent
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((x_latent.shape[0],), step, device=x_latent.device, dtype=torch.long)
            x_dec, _ = self.p_sample_ddim(x_dec, cond, ts, index=index, use_original_steps=use_original_steps,
                                          unconditional_guidance_scale=unconditional_guidance_scale,
                                          unconditional_conditioning=unconditional_conditioning)
        return x_dec



    def ddecode(self, x_latent, cond=None, t_start=50, temp = 1, unconditional_guidance_scale=1.0, unconditional_conditioning=None,
               use_original_steps=False):
        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        timesteps = timesteps[:t_start]

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='Decoding image', total=total_steps)
        x_dec = x_latent
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((x_latent.shape[0],), step, device=x_latent.device, dtype=torch.long)
            x_dec, _ = self.p_sample_ddim(x_dec, cond, ts, index=index, use_original_steps=use_original_steps, temperature = temp, 
                                          unconditional_guidance_scale=unconditional_guidance_scale,
                                          unconditional_conditioning=unconditional_conditioning)
        return x_dec
    

    def flow_matching_noise_injection(self, x0_y, t_n):
        """
        PnP-Flow style noise injection.
        
        Args:
            x0_y: measurement-informed estimate of x_0 (x_0|y)
            x_t: current latent
            sigma_noise: optional noise scale; if None, uses default schedule
        Returns:
            x_prev: resampled latent with flow-matching noise
        """
        device = self.model.device
        noise = torch.randn_like(x0_y, device=device)

        z_tilde_n = (1.0 - t_n) * noise + t_n * x0_y
        
        return z_tilde_n



               
