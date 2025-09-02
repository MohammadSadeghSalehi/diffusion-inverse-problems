# Diffusion for Inverse Problems

This repository provides **implementations of state-of-the-art diffusion-based methods for solving inverse problems** in imaging, including **DiffPIR, DPS, DDRM, and RePaint**.

<img src="notebooks/inpainting_reconstructed_diffpir.png" alt="inpainted" width="200"> <img src="notebooks/inpainting_reconstructed_dps.png" alt="inpainted" width="200">

---

## ✨ Highlights

- 🧩 **Modular algorithms**: easily swap between DDPM/DDIM pipelines
- 🖼️ **Plug-and-play operators** for inpainting, tomography (CT), and deblurring
- 📓 **Prebuilt Jupyter notebooks** for fast experimentation
- 🔬 **Real-world applications**: object removal (like Apple Retouch/Google Magic Eraser), CT reconstruction, and image deblurring

---

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/diffusion-inverse-problems.git
cd diffusion-inverse-problems

# Install dependencies
pip install -r requirements.txt

# (Optional) Install in editable mode for development
pip install -e .

# Run demo notebook
jupyter notebook notebooks/demo_inpainting.ipynb
```

---

## 📚 Implemented Algorithms

| Algorithm | Description | Paper |
|-----------|-------------|-------|
| **DiffPIR** | Proximal gradient with diffusion priors | [Zhu et al., 2023](https://arxiv.org/pdf/2305.08995) |
| **DPS** | Diffusion posterior sampling for noisy inverse problems | [Chung et al., 2023](https://openreview.net/forum?id=OnD9zGAGT0k) |
| **DDRM** | DDPM/DDIM schedulers for image restoration | [Kawar et al., 2022](https://arxiv.org/pdf/2201.11793) |
| **RePaint** | Iterative inpainting with resampling | [Lugmayr et al., 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Lugmayr_RePaint_Inpainting_Using_Denoising_Diffusion_Probabilistic_Models_CVPR_2022_paper.pdf) |

---

## 📦 Repository Structure

```
diffusion-inverse-problems/
│
├── algorithms/              # Core algorithm implementations
│   ├── diffpir.py          # DiffPIR implementation
│   ├── dps.py              # DPS implementation
│   ├── repaint.py          # RePaint implementation
│   ├── ddrm.py             # DDRM implementation
│   └── __init__.py
│
├── physics/                # Forward model operators
│   ├── inpainting.py       # Inpainting masks and operators
│   ├── tomography.py       # CT/MRI reconstruction operators
│   ├── blur.py             # Blur kernels and deblurring
│   └── __init__.py
│
├── utils/                  # Utility functions
│   ├── dataset.py          # Dataset loaders (MRI, CelebA-HQ, etc.)
│   ├── degrade.py          # Image corruption operators
│   ├── visualization.py    # Plotting and visualization tools
│   └── __init__.py
│
├── notebooks/              # Example notebooks and demos
│   ├── demo_inpainting.ipynb
│   ├── demo_ct.ipynb
│   └── demo_deblurring.ipynb
│
├── requirements.txt        # Python dependencies
├── README.md
└── LICENSE
```

---

## 📊 Example Usage

### Basic Inpainting Example

```python
from algorithms.repaint import RePaint
from physics.inpainting import InpaintingOperator
from utils.dataset import load_celeba_hq

# Load pretrained diffusion model and dataset
model = load_pretrained_model("celeba_hq_256")
image = load_celeba_hq()[0]

# Create inpainting mask and operator
mask = create_random_mask(image.shape, mask_ratio=0.3)
operator = InpaintingOperator(mask)

# Run RePaint algorithm
repaint = RePaint(model, operator)
restored_image = repaint.restore(image)
```

For complete examples, see the notebooks in `/notebooks/`.

---

## 🛠️ Supported Operations

- **Inpainting**: Remove unwanted objects, fill missing regions
- **CT Reconstruction**: Reconstruct medical images from sparse projections  
- **Deblurring**: Remove motion blur and camera shake
- **Super-resolution**: Enhance image resolution (coming soon)
- **MRI Reconstruction**: Reconstruct medical images from undersampled measurements (coming soon)

---

## 📋 Requirements

- Python 3.8+
- PyTorch 1.12+
- CUDA 11.0+ (for GPU acceleration)
- Additional dependencies in `requirements.txt`

---

## 🤝 Contributing

We welcome contributions! Please see our [contribution guidelines](CONTRIBUTING.md) for details on how to:

- Report bugs and request features
- Submit pull requests
- Add new algorithms or operators
- Improve documentation

---

## 📚 References

### Algorithm Papers

**RePaint: Inpainting using Denoising Diffusion Probabilistic Models**
```bibtex
@inproceedings{lugmayr2022repaint,
  title={RePaint: Inpainting using Denoising Diffusion Probabilistic Models},
  author={Lugmayr, Andreas and Danelljan, Martin and Romero, Andres and Yu, Fisher and Timofte, Radu and Van Gool, Luc},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={11461--11471},
  year={2022}
}
```

**Diffusion Posterior Sampling for General Noisy Inverse Problems**
```bibtex
@inproceedings{chung2023diffusion,
  title={Diffusion Posterior Sampling for General Noisy Inverse Problems},
  author={Chung, Hyungjin and Kim, Jeongsol and Mccann, Michael Thompson and Klasky, Marc Louis and Ye, Jong Chul},
  booktitle={The Eleventh International Conference on Learning Representations},
  year={2023},
  url={https://openreview.net/forum?id=OnD9zGAGT0k}
}
```

**Denoising Diffusion Models for Plug-and-Play Image Restoration**
```bibtex
@article{zhu2023denoising,
  title={Denoising Diffusion Models for Plug-and-Play Image Restoration},
  author={Zhu, Yuanzhi and Zhang, Kai and Liang, Jingyun and Cao, Jiezhang and Wen, Bihan and Timofte, Radu and Van Gool, Luc},
  journal={arXiv preprint arXiv:2305.08995},
  year={2023}
}
```

**Denoising Diffusion Restoration Models**
```bibtex
@article{kawar2022denoising,
  title={Denoising Diffusion Restoration Models},
  author={Kawar, Bahjat and Elad, Michael and Ermon, Stefano and Song, Jiaming},
  journal={arXiv preprint arXiv:2201.11793},
  year={2022}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Thanks to the authors of the original papers for their groundbreaking work
- Special thanks to the open-source community for various code contributions
- Built with ❤️ for the computer vision, inverse problems, and machine learning community
