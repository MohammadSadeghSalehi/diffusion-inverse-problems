from diffusers import DDPMScheduler
import torch
from tqdm import tqdm
def repaint(
    pipeline,
    corrupted_img: torch.Tensor,
    mask: torch.Tensor,
    num_inference_steps=250,
    jump_n_sample=5
):
    """
    Implementation of Algorithm 1 from RePaint paper.
    """
    device = corrupted_img.device
    batch_size = corrupted_img.shape[0]
    
    # Initialize scheduler and UNet
    scheduler = DDPMScheduler.from_config(pipeline.scheduler.config)
    scheduler.set_timesteps(num_inference_steps)
    unet = pipeline.unet.eval().to(device)

    # Create timestep-indexed buffers
    timesteps = scheduler.timesteps
    alphas = scheduler.alphas_cumprod.to(device)          # ᾱ_t
    betas = scheduler.betas.to(device)                    # β_t
    sqrt_alpha = torch.sqrt(alphas)
    sqrt_1m_alpha = torch.sqrt(1 - alphas)

    # Initialize x_T ~ N(0, I)
    x = torch.randn_like(corrupted_img).to(device)

    for i, t in enumerate(tqdm(timesteps, desc="RePaint")):
        for u in range(jump_n_sample):
            t_idx = t if isinstance(t, int) else t.item()
            beta_t = betas[t_idx]
            sqrt_alpha_bar = sqrt_alpha[t_idx]
            sqrt_one_minus_alpha_bar = sqrt_1m_alpha[t_idx]

            # ε ∼ N(0, I)
            noise = torch.randn_like(x) if t > 1 else torch.zeros_like(x)

            # x_known_t−1
            x_known_tminus1 = sqrt_alpha_bar * corrupted_img + sqrt_one_minus_alpha_bar * noise

            # z ∼ N(0, I)
            z = torch.randn_like(x) if t > 1 else torch.zeros_like(x)

            # Line 7: Predict ε_theta
            t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
            with torch.no_grad():
                eps_theta = unet(x, t_tensor).sample

            sigma_t = torch.sqrt(beta_t)

            x_unknown_tminus1 = (
                (1 / torch.sqrt(scheduler.alphas[t_idx])) * 
                (x - beta_t / sqrt_one_minus_alpha_bar * eps_theta) +
                sigma_t * z
            )

            # Combine known and unknown regions
            x_tminus1 = mask * x_known_tminus1 + (1 - mask) * x_unknown_tminus1

            # Resample x_t from x_t−1
            if u < jump_n_sample - 1 and t > 1:
                mean = torch.sqrt(1 - betas[t_idx - 1]) * x_tminus1
                std = torch.sqrt(betas[t_idx - 1])
                x = mean + std * torch.randn_like(x)
            else:
                x = x_tminus1.clone()

    return x.clamp(-1, 1)