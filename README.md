# mlegpu

GPU-accelerated (JAX) maximum-likelihood 3D localization of single molecules with a vectorial PSF model.

<p align="center">
  <img src="docs/logo.svg" alt="mlegpu" width="480">
</p>

Code accompanying:

*Three-dimensional single-molecule tracking in living cells with isotropic precision of 15 nm.*

**Anna Gaugutz<sup>1</sup>, Lukas Velas<sup>1,2</sup>, Elena Georgiou<sup>3</sup>, Julian Maloberti<sup>4</sup>, Simon Moser<sup>4</sup>, Alexander Jesacher<sup>4</sup>, Marina Bishara<sup>1</sup>, Johannes B. Huppa<sup>5,6,7</sup>, Stefan Howorka<sup>3,8,\*</sup>, Gerhard J. Schütz<sup>1,\*</sup>**

<sup>\*</sup> Corresponding authors 

## Requirements

- Linux or Windows via WSL2. GPU acceleration needs an NVIDIA GPU with CUDA.
- Python ≥ 3.10 and the packages in `requirements.txt`.

## Installation

```bash
git clone https://github.com/TODO/mlegpu.git
cd mlegpu
pip install -r requirements.txt
pip install -U "jax[cuda12]"   # GPU support (Linux / WSL2)
```

## Usage

See `notebooks/GPU_fit.ipynb`. In short:

```python
from SMLM import *

cam = Camera(pixsize=6500, QE=0.75, baseline=100, amp=0.24)
h = psf(NA=1.5, wavelength=670, RI=[1.33, 1.518], z_min=0, z_max=1000, uz=10)

prelocs, images = preloc_and_crop(stack, sigma=1, min_dist=7, N_crop=h.N_crop)
images = cts2phs(images, cam)
results, raw_results = perform_fit(images, prelocs, h)
results = add_CRLB(h, cam, results)
```

All lengths are in nanometres. Photon counts refer to photons in the back focal plane.

## Citation

Citation details will be added upon posting a preprint or publication of the accompanying manuscript.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.