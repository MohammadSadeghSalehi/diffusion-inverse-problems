import matplotlib.pyplot as plt
import torch


def show_images(images, titles=None, ncols=None, figsize=(12, 4), cmap=None, metrics=None, save_path=None):
    """
    Display list of images in a row.

    Args:
        images (list of torch.Tensor): List of tensors (C,H,W).
        titles (list of str): Optional titles for each image.
        ncols (int): Number of columns in grid.
        figsize (tuple): Figure size.
        cmap (str): Color map for grayscale.
    """
    # set font size
    plt.rcParams["font.size"] = 16
    num_images = len(images)
    if ncols is None:
        ncols = num_images
    nrows = (num_images + ncols - 1) // ncols

    plt.figure(figsize=figsize)
    for i, img in enumerate(images):
        plt.subplot(nrows, ncols, i + 1)
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu()
            if img.dim() == 3 and img.size(0) in [1, 3]:
                img = img.permute(1, 2, 0)
            elif img.dim() == 4:
                img = img.squeeze(0).permute(1, 2, 0)
        plt.imshow(img, cmap=cmap)
        if titles:
            plt.title(titles[i])
            if metrics:
                m_dict = metrics[i] if isinstance(metrics, list) else metrics
                for j, (metric_name, metric_value) in enumerate(m_dict.items()):
                    plt.text(
                        0.5, -0.1 - j * 0.08,
                        f"{metric_name}: {metric_value:.2f}",
                        ha="center", va="center",
                        transform=plt.gca().transAxes,
                        fontsize=14
                    )
        plt.axis("off")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.show()
    else:
        plt.show()


def save_image_grid(images, path, ncols=None):
    """
    Save a grid of images to disk.
    """
    num_images = len(images)
    if ncols is None:
        ncols = num_images
    nrows = (num_images + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3))
    axes = axes.flatten()
    for i, img in enumerate(images):
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().permute(1, 2, 0)
        axes[i].imshow(img)
        axes[i].axis("off")
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()