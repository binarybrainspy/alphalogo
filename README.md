# AlphaLogo

**Complex-Plane Parameterization and Geometric Analysis of Heptapod B**

AlphaLogo is a research project and open-source pipeline for the formal mathematical encoding of *Heptapod B* — the constructed circular written language from Denis Villeneuve's *Arrival* (2016). It establishes the first geometric framework for representing, comparing, and interpolating logograms in this language, and ships a companion website for public exploration.

---

## What This Project Does

Each Heptapod B logogram is a closed circular glyph encoding a complete sentence. AlphaLogo:

1. **Encodes** each logogram as a complex-valued parametric curve lifted into the complex plane
2. **Decomposes** the curve via Fourier analysis into a normalized descriptor vector **Φ(L)**
3. **Reconstructs** any logogram exactly from its Φ(L) encoding (invertible transform)
4. **Interpolates** between two logograms in Φ(L) space, producing structurally coherent intermediate symbols
5. **Evaluates** interpolation quality via a morphological smoothness score
6. **Publishes** an interactive website showcasing the 49 known logograms, their meanings, and the Fourier construction method live in-browser

The guiding analogy is the **Laplace transform**: just as it converts a differential equation into algebra in the *s*-domain — preserving all information and permitting an exact inverse — AlphaLogo converts a logogram into algebra in Φ(L)-space. Operations performed there map back to valid logograms via the inverse transform.

---

## Results (Track 1)

| Metric | Value |
|--------|-------|
| Logograms encoded | 49 / 49 |
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
├── logograms_wire/             # Skeletal Fourier reconstructions (for website)
├── website/
│   └── images             # AlphaLogo public website
├── heptapod_encode.py          # Core encoding/decoding module (v5)
├── interpolation.py            # Interpolation between two logograms
├── track1_grid.py              # Full interpolation grid study
├── generate_wireframes.py      # Generate logograms_wire/ from logograms/
├── getting_parameters.py       # Batch encode all logograms
├── phi_matrix.npy              # Encoded dataset (49 x 783 matrix)
├── index.html                  # website index page       
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
```bash
# Edit DATASET_FOLDER, IDX_A, IDX_B in interpolation.py
python interpolation.py
```

### Run the full grid study
```bash
python track1_grid.py
# Outputs: grid_summary.png, interpolation_scores.txt, best_pair_*.png
```

### Generate wireframe images for the website
```bash
python generate_wireframes.py --input logograms/ --output logograms_wire/
```

### Deploy the website
Drop website/index.html, logograms/, and logograms_wire/ into one folder.
Drag to Netlify or run `vercel` in the folder. No build step needed.

---

## Core API

```python
from heptapod_encode import (
    encode_logogram,      # Single PNG -> LogogramEncoding
    batch_encode,         # Folder of PNGs -> dict of encodings
    save_encodings,       # dict -> (49x783) numpy matrix
    reconstruct_curves,   # Phi(L) vector -> list of complex curves
    render_curves,        # complex curves -> numpy image
    align_phases,         # Phase-align Phi(B) to Phi(A) before interpolation
)
```

---

## Why Track 2 (Generative Modeling) Is Deferred

Generating genuinely new Heptapod B logograms is the most compelling long-term goal of this project. After experimenting with a beta-VAE and an interpolation-based generator, the conclusion is that **49 training samples is not enough** to produce novel logograms that are structurally distinct from the known ones.

**The specific failure modes:**

The VAE learned to produce closed blobs with rough circularity but no internal tendril structure — it approximated "roughly circular closed curve" but not "Heptapod B logogram." The interpolation generator produced mathematically valid intermediate logograms, but every candidate is a weighted blend of existing symbols — a hybrid, not a new word.

**What needs to happen before generative modeling is viable:**

1. **More data.** The full production vocabulary is approximately 71 logograms. The remaining ~22 need to be extracted from film frames or the production art book. At 71 samples the problem changes meaningfully.

2. **Stroke graph representation.** Decomposing each logogram into a graph of strokes (nodes = junction points, edges = individual strokes) and training a graph generative model (GraphRNN or similar) would capture the compositional grammar of how symbols are built.

3. **Topological data analysis.** Applying persistent homology extracts language-level structural features — number of loops, branch connectivity — independent of geometry. This gives a linguist-usable grammar before any generation happens.

4. **Diffusion model on Φ(L).** Once 71+ samples are available, a denoising diffusion model on Φ(L) vectors is more appropriate than a VAE in low-data regimes. See *Ho et al. (2020), Denoising Diffusion Probabilistic Models*.

5. **Linguist collaboration.** The right sequencing is: send Track 1 interpolation results to a constructed-language linguist, receive feedback on observable grammatical patterns, use those patterns to constrain generation rather than generating freely.

Track 2 is deferred, not abandoned. The Φ(L) encoding built in Track 1 is the right foundation — data is the only real blocker.

---

## Roadmap

- **Track 1** ✅ — Complex-plane parameterization and interpolation study
- **Track 2** ⏸  — Generative modeling (deferred — see above)
- **Track 3** ✅ — AlphaLogo public website

---

## Paper

> *AlphaLogo: Complex-Plane Parameterization of Heptapod B*
> BinaryBrains, 2025
> See `paper/*.pdf`

---

## Dataset

The 49 Heptapod B logograms are sourced from a publicly available Kaggle dataset derived from the film's production materials. Heptapod B was designed by linguist Jessica Coon and artist Martine Bertrand for *Arrival* (2016), dir. Denis Villeneuve.

---

## Contributing

- **Developers** — extend the encoding pipeline, improve the website, add analysis scripts
- **Linguists** — evaluate generated candidates, propose grammatical constraints
- **Data contributors** — help extract the remaining ~22 logograms from production materials

Open an issue or PR, or email `lucascollections31@gmail.com`.

---

## License

MIT License. See LICENSE for details.