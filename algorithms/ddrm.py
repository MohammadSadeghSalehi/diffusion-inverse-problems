from diffusers import DDIMScheduler, DDIMPipeline, DDPMScheduler, DDPMPipeline
import torch
from tqdm import tqdm

def ddrm_ddim(
    pipeline: DDIMPipeline,
    y: torch.Tensor,
    A: callable,
    A_pinv: callable,
    num_inference_steps=50,
    x_shape=None,
):
    """
    DDRM for general linear inverse problems using a DDIM scheduler.
    """
    scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    scheduler.set_timesteps(num_inference_steps)
    unet = pipeline.unet.eval()
    device = y.device
    
    # Start from pure noise with correct shape
    if x_shape is None:
        x_shape = y.shape  # Fallback to y shape if not specified
    x = torch.randn(x_shape).to(device)

    for i, t in enumerate(tqdm(scheduler.timesteps)):
        t_tensor = torch.full((x.shape[0],), t, device=device, dtype=torch.long)
        
        with torch.no_grad():
            # Predict noise from the current noisy sample x_t
            eps = unet(x, t_tensor).sample

            # Predict x0 using the DDIM formula
            alpha_bar_t = scheduler.alphas_cumprod[t]
            x0_pred = (x - torch.sqrt(1 - alpha_bar_t) * eps) / torch.sqrt(alpha_bar_t)

            # Apply Data Consistency projection
            x0_proj = x0_pred - A_pinv(A(x0_pred) - y)

            # Take the DDIM reverse step with the projected x0
            if i < len(scheduler.timesteps) - 1:
                prev_t = scheduler.timesteps[i + 1]
                alpha_bar_prev = scheduler.alphas_cumprod[prev_t]

                # Use the original eps for the next step (DDIM is deterministic)
                x = (torch.sqrt(alpha_bar_prev) * x0_proj) + (torch.sqrt(1 - alpha_bar_prev) * eps)
            else:
                # Final step: return projected x0
                x = x0_proj

    return x
def ddrm_ddpm(
    pipeline: DDPMPipeline,
    y: torch.Tensor,
    A: callable,
    A_pinv: callable,
    num_inference_steps=100,
    x_shape=None,
):
    """
    DDRM for general linear inverse problems using a DDPM scheduler.
    
    Key differences from DDIM version:
    1. Uses stochastic sampling (adds noise at each step)
    2. Follows DDPM posterior sampling formula
    3. More steps typically needed for good results
    """
    scheduler = DDPMScheduler.from_config(pipeline.scheduler.config)
    scheduler.set_timesteps(num_inference_steps)
    unet = pipeline.unet.eval()
    device = y.device

    # Get scheduler parameters
    alphas_cumprod = scheduler.alphas_cumprod.to(device)
    
    # Start from pure noise with correct shape
    if x_shape is None:
        x_shape = y.shape  # Fallback to y shape if not specified
    x = torch.randn(x_shape).to(device)

    for i, t in enumerate(tqdm(scheduler.timesteps)):
        t_tensor = torch.full((x.shape[0],), t, device=device, dtype=torch.long)
        
        with torch.no_grad():
            # Predict noise from the current noisy sample x_t
            eps = unet(x, t_tensor).sample

            # Predict x0 using the DDPM formula
            alpha_bar_t = alphas_cumprod[t]
            x0_pred = (x - torch.sqrt(1 - alpha_bar_t) * eps) / torch.sqrt(alpha_bar_t)

            # Apply Data Consistency projection
            x0_proj = x0_pred - A_pinv(A(x0_pred) - y)

            # DDPM reverse step with projected x0
            if i < len(scheduler.timesteps) - 1:
                # Get next timestep
                prev_timestep = scheduler.timesteps[i + 1]
                alpha_bar_prev = alphas_cumprod[prev_timestep]
                
                # Calculate mean of posterior distribution
                mean_prev = torch.sqrt(alpha_bar_prev) * x0_proj
                
                # Calculate variance for stochastic sampling
                variance = 1 - alpha_bar_prev
                
                if prev_timestep > 0:  # Add noise if not the final step
                    noise = torch.randn_like(x)
                    x = mean_prev + torch.sqrt(variance) * noise
                else:
                    x = mean_prev  # Final step: no noise
            else:
                # Final iteration: return projected x0
                x = x0_proj

    return x