import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from datasets import load_dataset
from diffusers import DDPMPipeline
from torch.utils.data import TensorDataset
import os
import requests
from tqdm import tqdm
import numpy as np
from PIL import Image


def download_file(url, local_path):
    """Download a file from URL with a progress bar."""
    if os.path.exists(local_path):
        print(f"✅ Found existing file at {local_path}")
        return local_path

    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    print(f"⬇️ Downloading dataset from {url} ...")

    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024

    with open(local_path, "wb") as f, tqdm(
        desc="Downloading",
        total=total_size,
        unit="iB",
        unit_scale=True,
    ) as bar:
        for data in response.iter_content(block_size):
            f.write(data)
            bar.update(len(data))

    print(f"✅ Download complete: {local_path}")
    return local_path


def load_ct_dataset(local_path, img_size, subset_ratio):
    data = torch.load(local_path)["train_imgs"]  # [N, H, W] or [N, 1, H, W]
    if data.ndim == 3:
        data = data.unsqueeze(1)

    transform = transforms.Resize((img_size, img_size))
    tensor_list = []
    for i in range(int(len(data) * subset_ratio)):
        img = data[i].float()
        img = transform(img)
        img = img.repeat(3, 1, 1)  # grayscale → 3 channels
        tensor_list.append(img)

    return TensorDataset(torch.stack(tensor_list))

def load_from_huggingface(id = "PranomVignesh/MRI-Images-of-Brain-Tumor", batch_size=16, img_size=224, subset_ratio=1.0, test = False):
    """
    Loads Hugging Face datasets and returns a PyTorch DataLoader

    Args:
        batch_size (int): The number of images per batch.
        img_size (int): The target size for resizing the images.
        subset_ratio (float): The ratio of the dataset to use.

    Returns:
        torch.utils.data.DataLoader: A DataLoader for the dataset.
    """
    print("Loading the dataset from Hugging Face...")
    # Load the dataset from Hugging Face.
    dataset = load_dataset(id, split="train")
    if test:
        dataset = load_dataset(id, split="test")

    # Subset the dataset if needed
    total_size = len(dataset)
    subset_size = int(total_size * subset_ratio)
    indices = np.random.permutation(total_size)[:subset_size]
    dataset = dataset.select(indices)
    print(f"Loaded {len(dataset)} samples from the dataset.")
    # Define a set of transformations to prepare the images.
    preprocess = transforms.Compose([
        transforms.CenterCrop((img_size, img_size)),
        transforms.ToTensor(), # Converts PIL Image to a PyTorch tensor (C, H, W)
        transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x), # Ensure 3 channels
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Normalize to [-1, 1]
    ])
    
    def transform_function(examples):
        # We process the 'image' column from the dataset
        images = [Image.fromarray(np.array(img)).convert('RGB') if isinstance(img, list) else img.convert('RGB')
                  for img in examples['image']]
        transformed_images = [preprocess(img) for img in images]
        # Return a dictionary with the transformed images
        return {'image': transformed_images}

    # Use the `with_transform` method to apply our function
    # This correctly handles the batching behavior of the datasets library.
    dataset.set_transform(transform_function)

    print("\nStep 3: Creating a DataLoader to handle batches...")
    # Create a DataLoader. No custom collate_fn is needed because the `with_transform`
    # handles the image to tensor conversion.
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False, # Shuffle is set to False for consistency with OOD testing
    )
    
    return dataloader

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
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
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

    elif dataset_name.lower() in ["ct"]:
        # === Auto-download LoDoPaB dataset ===
        url = "https://huggingface.co/datasets/deepinv/LoDoPaB-CT_toy/resolve/main/LoDoPaB-CT_small.pt?download=true"
        local_path = os.path.join("data", "LoDoPaB-CT_small.pt")
        local_path = download_file(url, local_path)

        dataset = load_ct_dataset(local_path, img_size, subset_ratio)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    elif dataset_name.lower() in ["bsds", "bsds500"]:
        try:
            bsds_loader = load_from_huggingface(id="aseeransari/BSDS", batch_size=batch_size, img_size=img_size, subset_ratio=subset_ratio)
            print("✓ BSDS500 loaded successfully")
            dataloader = bsds_loader
        except Exception as e:
            print(f"✗ BSDS500 not available: {e}")
            dataloader = None
    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")

    return pipeline, dataloader