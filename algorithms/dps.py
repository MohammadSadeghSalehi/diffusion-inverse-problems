from diffusers import DDIMScheduler, DDIMPipeline, DDPMScheduler, DDPMPipeline
import matplotlib.pyplot as plt
import torch 
from tqdm import tqdm

def dps(
    pipeline,
    y: torch.Tensor,
    A: callable,
    A_pinv: callable,
    num_steps=50,
    eta=0.0,
    zeta=1.0,
    gt=None,
    verbose=False
):
    """
    DPS-DDPM.
    
    Args:
        pipeline: Diffusion pipeline with UNet and scheduler
        y: Measurements
        A: Forward physics operator
        A_pinv: Pseudo-inverse operator
        num_steps: Number of diffusion steps
        eta: DDIM stochasticity parameter
        zeta: Data fidelity gradient weight
        gt: Ground truth for evaluation
    """
    device = y.device
    unet = pipeline.unet.to(device)
    scheduler = pipeline.scheduler

    # Scheduler setup
    num_train_timesteps = scheduler.config.num_train_timesteps
    betas = scheduler.betas.to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    # Timestep schedule (from T to 0)
    skip = num_train_timesteps // num_steps
    time_steps = list(reversed(range(0, num_train_timesteps, skip)))

    # Initialize from pure noise (standard DDPM initialization)
    x = torch.randn_like(A_pinv(y))
    
    psnr_history = []
    
    # Main DPS-DDPM loop
    for i, t in enumerate(tqdm(time_steps, desc="DPS")):
        xt = x.detach()
        
        at = alphas_cumprod[t]
        at_next = alphas_cumprod[time_steps[i + 1]] if i + 1 < len(time_steps) else torch.tensor(1.0, device=device)
        bt = 1.0 - at / at_next

        # Forward pass with gradient computation for data fidelity
        with torch.enable_grad():
            xt_grad = xt.detach().requires_grad_()
            t_tensor = torch.full((xt.shape[0],), t, device=device, dtype=torch.long)
            
            # UNet prediction
            eps_pred = unet(xt_grad, t_tensor).sample
            x0_t = (xt_grad - eps_pred * (1 - at).sqrt()) / at.sqrt()
            x0_t = torch.clamp(x0_t, -1.0, 1.0)
            
            # Data fidelity loss and gradient
            data_loss = 0.5 * torch.norm(A(x0_t) - y, p=2) ** 2
            grad = torch.autograd.grad(data_loss, xt_grad)[0].detach()
        
        
        # DDPM/DDIM sampling step
        epsilon = torch.randn_like(xt)
        sigma_tilde = ((bt * (1 - at_next)) / (1 - at)).sqrt() * eta
        c2 = ((1 - at_next) - sigma_tilde**2).sqrt()

        # Update x for next iteration
        x = (
            at_next.sqrt() * x0_t.detach()
            + c2 * eps_pred.detach()
            + sigma_tilde * epsilon
            - zeta * at_next.sqrt() * grad  # Scaled gradient for data consistency
        )
        
        # Track progress
        if verbose and (i % 10 == 0):
            with torch.no_grad():
                if gt is not None:
                    mse = torch.mean(((x0_t + 1) / 2 - (gt + 1) / 2) ** 2)
                    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse + 1e-8))
                    psnr_history.append(psnr.item())
                    print(f"Step {i}, t={t}: PSNR = {psnr.item():.2f} dB")
                
                if i % 20 == 0:  # Show image less frequently
                    plt.figure(figsize=(8, 4))
                    plt.subplot(1, 2, 1)
                    plt.imshow(((x0_t[0] + 1) / 2).detach().cpu().permute(1, 2, 0).numpy())
                    plt.title(f"x0 estimate at t={t}")
                    plt.axis("off")
                    
                    plt.subplot(1, 2, 2)
                    plt.imshow(((A_pinv(y)[0] + 1) / 2).detach().cpu().permute(1, 2, 0).numpy())
                    plt.title("Pseudo-inverse")
                    plt.axis("off")
                    plt.tight_layout()
                    plt.show()

    # Final denoising
    with torch.no_grad():
        t_tensor = torch.zeros((x.shape[0],), device=device, dtype=torch.long)
        eps_pred = unet(x, t_tensor).sample
        x_final = (x - eps_pred * (1 - alphas_cumprod[0]).sqrt()) / alphas_cumprod[0].sqrt()
        x_final = torch.clamp(x_final, -1.0, 1.0)

    return x_final