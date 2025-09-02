import torch
import numpy as np
from torchvision import transforms
from torch.nn.functional import interpolate
from skimage.metrics import structural_similarity as ssim_sk
from skimage.metrics import peak_signal_noise_ratio as psnr_sk
from torchmetrics.image.fid import FrechetInceptionDistance
import lpips


class MetricsEvaluator:
    def __init__(self, device=None, lpips_net="alex"):
        self.device = device or ("cuda" if torch.cuda.is_available()
                                 else "mps" if torch.backends.mps.is_available()
                                 else "cpu")

        # Initialize LPIPS (once)
        self.lpips_loss = lpips.LPIPS(net=lpips_net).to("cpu" if self.device == "mps" else self.device)

        # Initialize FID (once)
        self.fid_metric = FrechetInceptionDistance(normalize=True).to(torch.float32).to("cpu" if self.device.type == "mps" else self.device)

        # Normalizer (for FID/Inception)
        self.normalize = transforms.Normalize([0.5] * 3, [0.5] * 3)

    # -------------------------------
    # Preprocessing
    # -------------------------------
    def preprocess(self, imgs, size=299):
        """
        Resize and normalize images to match Inception requirements.
        Input: imgs in [0,1], shape (B,C,H,W).
        """
        imgs_resized = interpolate(imgs, size=(size, size),
                                   mode='bilinear', align_corners=False)
        return self.normalize(imgs_resized)

    # -------------------------------
    # PSNR & SSIM
    # -------------------------------
    def psnr_ssim(self, preds, targets):
        preds_np = preds.clamp(0, 1).cpu().numpy()
        targets_np = targets.clamp(0, 1).cpu().numpy()
        psnr_vals, ssim_vals = [], []

        for pred, gt in zip(preds_np, targets_np):
            pred_img = np.transpose(pred, (1, 2, 0))
            gt_img = np.transpose(gt, (1, 2, 0))

            psnr_vals.append(psnr_sk(gt_img, pred_img, data_range=1.0))
            ssim_vals.append(ssim_sk(gt_img, pred_img, data_range=1.0, channel_axis=-1))

        return np.mean(psnr_vals), np.mean(ssim_vals)

    # -------------------------------
    # LPIPS
    # -------------------------------
    def lpips_score(self, preds, targets):
        device = "cpu" if self.device == "mps" else self.device
        preds, targets = preds.to(device), targets.to(device)
        lpips_vals = []
        for p, t in zip(preds, targets):
            lp = self.lpips_loss(p.unsqueeze(0), t.unsqueeze(0))
            lpips_vals.append(lp.item())
        return np.mean(lpips_vals)

    # -------------------------------
    # FID
    # -------------------------------
    def fid_score(self, preds, targets):
        # Ensure correct dtype & range
        preds = preds.clamp(0, 1).to(torch.float32)
        targets = targets.clamp(0, 1).to(torch.float32)

        # Handle device incompatibility (MPS → CPU)
        if preds.device.type == "mps" or targets.device.type == "mps":
            preds = preds.to("cpu")
            targets = targets.to("cpu")

        if preds.shape[0] == 1 and targets.shape[0] == 1:  # FID needs 2 samples
            preds = torch.cat([preds]*2)
            targets = torch.cat([targets]*2)
        # Check batch shape
        if preds.ndim == 3:
            preds = preds.unsqueeze(0)  # (3,H,W) → (1,3,H,W)
        if targets.ndim == 3:
            targets = targets.unsqueeze(0)

        # Reset metric state
        self.fid_metric.reset()

        # Update with real and fake samples
        self.fid_metric.update(targets, real=True)
        self.fid_metric.update(preds, real=False)

        return float(self.fid_metric.compute())

    # -------------------------------
    # Full evaluation
    # -------------------------------
    def compute_all(self, preds, targets):
        psnr_val, ssim_val = self.psnr_ssim(preds, targets)
        lpips_val = self.lpips_score(preds, targets)
        fid_val = self.fid_score(preds, targets)

        return {
            "PSNR": psnr_val,
            "SSIM": ssim_val,
            "LPIPS": lpips_val,
            "FID": fid_val,
        }