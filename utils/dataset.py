import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from datasets import load_dataset
from diffusers import DDPMPipeline


def get_pipeline_and_dataset(
    model_name="google/ddpm-celebahq-256",
    dataset_name="CelebA-HQ",
    batch_size=1,
    img_size=96,
    subset_ratio=0.01,
    device="cpu"
):
    """
    Load diffusion pipeline and dataset.

    Args:
        model_name (str): HuggingFace model ID for pretrained diffusion model.
        dataset_name (str): Dataset name ("MRI", "CelebA-HQ", etc.)
        batch_size (int): Batch size for dataloader.
        img_size (int): Image resolution.
        subset_ratio (float): Fraction of dataset to use.
        device (str): "cuda", "mps", or "cpu".

    Returns:
        pipeline: DDPMPipeline
        dataloader: torch.utils.data.DataLoader
    """
    # Load pretrained diffusion model
    pipeline = DDPMPipeline.from_pretrained(model_name).to(device)

    # Preprocessing
    transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.CenterCrop(img_size),
        T.ToTensor(),
    ])

    if dataset_name.lower() in ["celeba-hq", "celeba"]:
        dataset = load_dataset("huggan/CelebA-HQ", split="train")
        dataset = dataset.shuffle(seed=42).select(
            range(int(len(dataset) * subset_ratio))
        )

        def preprocess(example):
            img = transform(example["image"])
            return {"image": img}
        
        dataset.set_transform(lambda x: preprocess(x))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    elif dataset_name.lower() == "mri":
        dataset = load_dataset("fastmri", "multicoil_knee", split="train")
        dataset = dataset.shuffle(seed=42).select(
            range(int(len(dataset) * subset_ratio))
        )

        def preprocess(example):
            img = example["image"].convert("L")  # grayscale MRI
            img = transform(img).repeat(3, 1, 1)  # convert to 3 channels
            return {"image": img}
        
        dataset.set_transform(lambda x: preprocess(x))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")

    return pipeline, dataloader