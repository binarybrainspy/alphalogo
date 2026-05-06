# AlphaLogo

**Complex-Plane Parameterization and Geometric Analysis of Heptapod B**

AlphaLogo is a research project and open-source pipeline for the formal mathematical encoding of *Heptapod B* — the constructed circular written language from Denis Villeneuve's *Arrival* (2016). It establishes the first geometric framework for representing, comparing, interpolating, and eventually generating logograms from this language.

---

## What This Project Does

Each Heptapod B logogram is a closed circular glyph encoding a complete sentence. AlphaLogo:

1. **Encodes** each logogram as a complex-valued parametric curve lifted into the complex plane
2. **Decomposes** the curve via Fourier analysis into a normalized descriptor vector **Φ(L)**
3. **Reconstructs** any logogram exactly from its Φ(L) encoding (invertible transform)
4. **Interpolates** between two logograms in Φ(L) space, producing structurally coherent intermediate symbols
5. **Evaluates** interpolation quality via a morphological smoothness score

The Laplace transform analogy: just as the Laplace transform converts a differential equation into algebra in the *s*-domain, AlphaLogo converts a logogram into algebra in Φ(L)-space. Operations there map back to valid logograms via the inverse transform.

---

## Results (Track 1)

Interpolation study across 25 logogram pairs drawn from 49 known symbols:

| Metric | Value |
|--------|-------|
| Pairs tested | 25 |
| Pairs scoring > 0.85 | 20 (80%) |
| Top smoothness score | 0.9517 (logograms 30 → 39) |
| Mean smoothness score | 0.804 |
| Φ(L) dimensionality | 783 (K=3 curves, N=64 harmonics) |

---

## Project Structure

```
alphalogo/
├── logograms/                  # 49 Heptapod B PNG images (dataset)
├── heptapod_encode.py          # Core encoding/decoding module (v5)
├── interpolation.py            # Interpolation between two logograms
├── track1_grid.py              # Full interpolation grid study
├── getting_parameters.py       # Batch encode all logograms → phi_matrix.npy
├── phi_matrix.npy              # Encoded dataset (49 × 783 matrix)
├── results/
│   ├── interpolation_scores.txt
│   ├── grid_summary.png
│   └── best_pair_*.png
├── paper/
│   └── AlphaLogo v1.0.pdf
├── CHANGELOG.md
└── README.md
```

---

## Installation

```bash
git clone https://github.com/binarybrainspy/alphalogo.git
cd alphalogo
pip install opencv-python scikit-image numpy matplotlib scipy
```

---

## Quick Start

### Encode all logograms
```python
from heptapod_encode import batch_encode, save_encodings

encodings = batch_encode("logograms/")
matrix, names = save_encodings(encodings, "phi_matrix.npy")
# matrix shape: (49, 783)
```

### Interpolate between two logograms
```python
# Edit interpolation.py:
#   DATASET_FOLDER = "logograms/"
#   IDX_A = 16
#   IDX_B = 30
python interpolation.py
```

### Run the full grid study
```python
python track1_grid.py
# Outputs: grid_summary.png, interpolation_scores.txt, best_pair_*.png
```

---

## Core API

```python
from heptapod_encode import (
    encode_logogram,      # Single PNG → LogogramEncoding
    batch_encode,         # Folder of PNGs → dict of encodings
    save_encodings,       # dict → (49×783) numpy matrix
    reconstruct_curves,   # Φ(L) vector → list of complex curves
    render_curves,        # complex curves → 224×224 numpy image
    align_phases,         # Phase-align Φ(B) to Φ(A) before interpolation
)
```

---

## Roadmap

- **Track 1** ✅ — Complex-plane parameterization and interpolation study
- **Track 2** 🔜 — Variational Autoencoder on Φ(L) for new logogram generation
- **Track 3** 🔜 — Interactive 3D sphere visualization website

---

## Paper

> *AlphaLogo: Complex-Plane Parameterization of Heptapod B — A Geometric Framework for Encoding, Interpolating, and Generating a Constructed Circular Language*
> Job Lucas, 2025

See `paper/AlphaLogo v1.0.docx` for the full research paper.

---

## Dataset

The 49 Heptapod B logograms used in this study are sourced from a publicly available Kaggle dataset derived from the film's production materials.

---

## License

MIT License. See LICENSE for details.
