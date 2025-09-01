import torch
from torch.utils.data import DataLoader


def degrade_dataloader(dataloader, A, noise_std=0.0, device="cpu"):
    """
    Apply forward physics operator + Gaussian noise to dataloader samples.

    Args:
        dataloader: original dataloader yielding clean images
        A (callable): forward operator
        noise_std (float): Gaussian noise std
        device (str): device

    Returns:
        DataLoader yielding corrupted samples
    """
    corrupted_data = []
    for batch in dataloader:
        x = batch["image"].to(device) if isinstance(batch, dict) else batch[0].to(device)
        y = A(x)
        if noise_std > 0:
            y += noise_std * torch.randn_like(y)
        corrupted_data.append(y.detach().cpu())

    corrupted_tensor = torch.cat(corrupted_data, dim=0)
    corrupted_dataset = torch.utils.data.TensorDataset(corrupted_tensor)
    return DataLoader(corrupted_dataset, batch_size=dataloader.batch_size, shuffle=False)