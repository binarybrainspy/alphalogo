# Changelog

All notable changes to the AlphaLogo project are documented here.

---

## [0.1.0] — Track 1 Complete — 2025

### Summary
First complete milestone of the AlphaLogo project. Establishes the full mathematical pipeline for encoding, reconstructing, and interpolating Heptapod B logograms using complex-plane Fourier parameterization.

---

### Added

#### Core Module — `heptapod_encode.py` (v5)

- **`_preprocess(img_path)`** — Loads PNG, applies binary threshold, extracts medial axis skeleton via Zhang-Suen skeletonization
- **`_extract_contours(skeleton, min_points)`** — Extracts structurally significant contours from skeleton, sorted by length
- **`_canonical_start(gamma)`** — Rolls each contour to a canonical starting point (rightmost point) for consistent phase reference across all logograms
- **`_encode_curve(contour, N, index)`** — Lifts a contour to the complex plane, centers it, applies DFT, normalizes by |c₁|, truncates/pads to exactly 2N+1 = 129 coefficients. Stores raw pixel `mean_pos` and `scale_factor` for exact reconstruction
- **`_build_phi(curves, max_curves)`** — Assembles the flat Φ(L) vector with fixed layout: `[real block | imag block | mean_pos_real | mean_pos_imag | scale_factors]`. Zero-pads unused curve slots so all logograms map to the same dimension
- **`encode_logogram(img_path)`** — Full single-logogram pipeline: PNG → `LogogramEncoding`
- **`batch_encode(folder_path)`** — Encodes all PNGs in a folder, rebuilds all Φ(L) vectors with the global `max_curves` so all are the same dimension
- **`save_encodings(encodings, output_path)`** — Stacks all Φ(L) vectors into an `(n_logograms × dim)` numpy matrix and saves as `.npy`
- **`reconstruct_curves(phi, N_harmonics, max_curves)`** — Inverse transform: Φ(L) vector → list of complex curves in pixel space. Skips zero-padded curve slots
- **`render_curves(curves, img_size, thickness, padding)`** — Draws reconstructed curves onto a white canvas with **auto-fit scaling**: computes global bounding box across all curves and scales to fill canvas with padding. Eliminates clipping artifacts on any coordinate range
- **`align_phases(phi_a, phi_b, N_harmonics, max_curves)`** — Phase-aligns Φ(B) to Φ(A) by computing the phase difference of the fundamental harmonic c₁ per curve and rotating B's descriptors accordingly. Prevents destructive Fourier interference during interpolation

#### Scripts

- **`getting_parameters.py`** — One-shot script to encode all 49 logograms and save the matrix
- **`interpolation.py` (v5)** — Interpolation between any two logograms with phase alignment and auto-fit rendering. Produces a visualization strip with original, reconstruction check, interpolated frames, and originals
- **`track1_grid.py`** — Full interpolation grid study across curated + random pairs with geodesic energy normalization, smoothness scoring, ranked output, and summary grid visualization

#### Results

- **49/49 logograms** encoded successfully into Φ(L) space
- **Φ(L) dimensionality**: 783 (K=3 max curves, N=64 harmonics)
- **25 logogram pairs** tested in the interpolation grid study
- **80% of pairs** scored above 0.85 morphological smoothness
- **Top pair**: logograms 30→39 with smoothness score 0.9517
- **Mean score**: 0.804 across all 25 pairs

#### Documents

- `README.md` — Project overview, installation, quick start, API reference
- `CHANGELOG.md` — This file
- `paper/AlphaLogo_paper.docx` — Full research paper (Track 1 results)

---

### Fixed (during Track 1 development)

- **v1→v2**: `_build_phi` used post-hoc zero-padding rather than pre-allocated section layout, causing `reconstruct_curves` to read scale/mean_pos values from the wrong byte positions → all reconstructions gave 0 curves
- **v2→v3**: `encode_logogram` stored `mean_pos` and `scale_factor` after accidental `img_size` division (224), causing reconstruction to place curves at ~0.4px instead of ~100px
- **v3→v4**: Added `_canonical_start` phase normalization to address destructive Fourier interference at interpolation midpoints
- **v4→v5**: `render_curves` clipped coordinates to `[0, 224]`, causing partial/invisible reconstructions for curves with any coordinate overflow. Replaced with auto-fit scaling to bounding box
- **Short contour fix**: Logograms with skeleton contours shorter than 2N+1=129 points (e.g. logogram 3900.png with 36 points) caused `ValueError` during DFT truncation. Fixed with symmetric zero-padding within the descriptor array

---

## [0.2.0] — Track 3 Complete — 2026

### Summary
AlphaLogo public website shipped. All 49 logograms with meanings, live Fourier
construction demo, morphing strip animation, and skeletal wireframe viewer.
Track 2 (generative modeling) formally deferred with documented rationale.

---

### Added

#### Website — `index.html`

#### Script — `generate_wireframes.py`
- Encodes every PNG in the logograms folder and renders its Fourier skeleton
  reconstruction as a 512×512 brushstroke image
- Output files named identically to inputs (000.png through 048.png) in
  a `logograms_wire/` folder
- Used by the website modal's "Skeletal" tab

#### Documentation
- `DEPLOY.md` — step-by-step Vercel and Netlify deployment guide
- `README.md` updated: roadmap, project structure, Track 2 deferral rationale,
  wireframe generation instructions

---

### Decisions

#### Track 2 — Generative Modeling — Formally Deferred
Attempted two approaches to generating new Heptapod B logograms:

1. **Beta-VAE on Φ(L)** — trained on 49 × 783 matrix with geometry-preserving
   augmentation to 500 samples. Generated candidates were circular but lacked
   tendril structure. Root cause: 49 samples is insufficient for a VAE to learn
   the structural grammar of the language, not an architecture problem.

2. **Interpolation-based generator** — pairwise paths, centroid sampling, and
   extrapolation across all logogram pairs. Produced geometrically valid candidates
   but every output is a blend of existing symbols by definition.

Decision: defer Track 2 until either (a) the remaining ~22 logograms from the
full production vocabulary are extracted, or (b) a linguist collaborator can
provide grammatical constraints to guide generation. See README for the full
technical rationale and five recommended approaches.

---