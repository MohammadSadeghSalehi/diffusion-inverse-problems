# Diffusion for Inverse Problems

This repository provides **implementations of state-of-the-art diffusion-based methods for solving inverse problems** in imaging, including **DiffPIR, DPS, DDRM, and RePaint**.

---

## ✨ Highlights

- 🧩 Modular algorithms: easily swap between **DDPM / DDIM pipelines**
- 🖼️ Plug-and-play operators for **inpainting, deblurring, tomography (CT)**
- 📓 Prebuilt **Jupyter notebooks** for fast experimentation
- 🔬 Applications: **object removal** (like Apple/Google), **CT reconstruction**, **deblurring**

---

## 🚀 Quickstart

```bash
# Clone the repository
git clone https://github.com/yourusername/diffusion-inverse-problems.git
cd diffusion-inverse-problems

# Install requirements
pip install -r requirements.txt

# (Optional) install in editable mode
pip install -e .

# Run demo notebook
jupyter notebook notebooks/demo_inpainting.ipynb
