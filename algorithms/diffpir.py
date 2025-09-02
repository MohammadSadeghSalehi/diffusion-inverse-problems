from diffusers import DDIMScheduler, DDIMPipeline
import torch
from tqdm import tqdm
def diffpir(
    pipeline,
    y: torch.Tensor,
    A: callable,
    A_pinv: callable,
    sigma_n: float,
    lambda_param: float = 7.0,
    zeta: float = 0.5,
    num_inference_steps=50,
    prox_iter=20,  # Number of iterations for the proximal subproblem (needs tuning)
    prox_stepsize=0.1, # Step size for proximal gradient descent (needs tuning)
    random_init=True
):
    device = y.device
    scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    scheduler.set_timesteps(num_inference_steps)
    unet = pipeline.unet.eval().to(device)

    timesteps = scheduler.timesteps
    if random_init:
        x = torch.randn((y.shape[0], 3, y.shape[2], y.shape[3]), device=device)
    else:
        x0_init = A_pinv(y)
        # Add noise scaled to match the first timestep's sigma
        first_t = scheduler.timesteps[0]  # Usually the largest timestep
        alpha_bar_0 = scheduler.alphas_cumprod[first_t].item()
        sigma_0 = ((1.0 - alpha_bar_0) / alpha_bar_0) ** 0.5  # DDPM noise level
        x = x0_init + sigma_0 * torch.randn_like(x0_init)

    for i, t in enumerate(tqdm(timesteps, desc="DiffPIR")):
        t_tensor = torch.full((x.shape[0],), t, device=device, dtype=torch.long)

        with torch.no_grad():
            eps_pred = unet(x, t_tensor).sample

        alpha_bar_t = scheduler.alphas_cumprod[t]
        alpha_bar_t_tensor = alpha_bar_t.clone().detach().to(device)
        sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t_tensor)
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - alpha_bar_t_tensor)

        # Predict x0
        x0_pred = (x - sqrt_one_minus_alpha_bar_t * eps_pred) / sqrt_alpha_bar_t
        x0_pred = x0_pred.clamp(-1, 1)

        # Data consistency (proximal subproblem)
        # Solve argmin_{x0} (1/2)||y - A(x0)||^2 + (lambda_param/2)||x0 - x0_pred||^2
        # using a simple iterative proximal gradient descent. For linear inverse problems it has a closed form solution
        
        rho_t = lambda_param * (sigma_n ** 2) / (alpha_bar_t + 1e-8)
        x0_corrected = x0_pred.clone().detach() # Initialize with the prediction

        for _ in range(prox_iter):
            # Gradient of the data-fidelity term: grad_x0( (1/2)||y - A(x0)||^2 ) = -A_pinv(y - A(x0))
            grad = -A_pinv(y - A(x0_corrected))
            x0_corrected = (x0_corrected - prox_stepsize * grad + rho_t * x0_pred) / (1 + rho_t)
            x0_corrected = x0_corrected.clamp(-1, 1)

        # Estimate eps_hat
        eps_hat = (x - sqrt_alpha_bar_t * x0_corrected) / sqrt_one_minus_alpha_bar_t

        # Add noise
        noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)

        # Compute next step x_{t-1}
        if i < len(timesteps) - 1:
            next_t = timesteps[i + 1]
            alpha_bar_next = scheduler.alphas_cumprod[next_t]
        else:
            alpha_bar_next = 1.0

        alpha_bar_next_tensor = torch.tensor(alpha_bar_next, device=device)
        sqrt_alpha_bar_next = torch.sqrt(alpha_bar_next_tensor)
        sqrt_one_minus_alpha_bar_next = torch.sqrt(1.0 - alpha_bar_next_tensor)

        x = (
            sqrt_alpha_bar_next * x0_corrected +
            sqrt_one_minus_alpha_bar_next * (
                torch.sqrt(torch.tensor(1.0 - zeta, device=device)) * eps_hat +
                torch.sqrt(torch.tensor(zeta, device=device)) * noise
            )
        )

    return x0_corrected.clamp(-1, 1)