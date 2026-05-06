"""
interpolation.py  (v5)
=======================
Logogram interpolation in Phi(L) space with phase alignment.
Requires heptapod_encode.py v5.

Usage:
    1. Set DATASET_FOLDER to your 49 logograms folder
    2. Set IDX_A and IDX_B (0-48)
    3. Run: python interpolation.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2
from pathlib import Path
from heptapod_encode import (
    batch_encode, save_encodings,
    reconstruct_curves, render_curves, align_phases,
)

# ── CONFIG ─────────────────────────────────────────────────────────────────────
DATASET_FOLDER = "logograms"          # <- folder with 49 logogram PNGs
PHI_MATRIX     = "phi_matrix.npy"
N_HARMONICS    = 64
STEPS          = 9
IDX_A          = 16
IDX_B          = 25

BG     = '#0d0d0d'
CYAN   = '#00d4ff'
ORANGE = '#ff6b35'
WHITE  = '#f0f0f0'
GRAY   = '#aaaaaa'

# ── ENCODE ─────────────────────────────────────────────────────────────────────
print("Encoding logograms...")
encodings     = batch_encode(DATASET_FOLDER, N_harmonics=N_HARMONICS)
matrix, names = save_encodings(encodings, PHI_MATRIX)
max_curves    = max(len(encodings[n].curves) for n in names)

print(f"\nMatrix     : {matrix.shape}")
print(f"max_curves : {max_curves}")

# ── VALIDATE ───────────────────────────────────────────────────────────────────
assert IDX_A < len(names), f"IDX_A={IDX_A} must be < {len(names)}"
assert IDX_B < len(names), f"IDX_B={IDX_B} must be < {len(names)}"
assert IDX_A != IDX_B,     "IDX_A and IDX_B must differ"

phi_a   = matrix[IDX_A]
phi_b   = matrix[IDX_B]

# Phase-align B to A before interpolating — prevents Fourier cancellation
phi_b_aligned = align_phases(phi_a, phi_b, N_HARMONICS, max_curves)

img_a = cv2.imread(str(Path(DATASET_FOLDER) / names[IDX_A]), cv2.IMREAD_GRAYSCALE)
img_b = cv2.imread(str(Path(DATASET_FOLDER) / names[IDX_B]), cv2.IMREAD_GRAYSCALE)

# ── SANITY CHECK ───────────────────────────────────────────────────────────────
print(f"\nSanity check...")
curves_a = reconstruct_curves(phi_a,         N_HARMONICS, max_curves)
curves_b = reconstruct_curves(phi_b_aligned, N_HARMONICS, max_curves)
recon_a  = render_curves(curves_a)
recon_b  = render_curves(curves_b)

print(f"  A ({names[IDX_A]}): {len(curves_a)} curves reconstructed")
print(f"  B ({names[IDX_B]}): {len(curves_b)} curves reconstructed")

if not curves_a:
    raise RuntimeError(f"Logogram A (idx={IDX_A}) gave 0 curves. "
                       "Check you are using heptapod_encode v5.")
if not curves_b:
    raise RuntimeError(f"Logogram B (idx={IDX_B}) gave 0 curves. "
                       "Check you are using heptapod_encode v5.")

xa = (min(c.real.min() for c in curves_a), max(c.real.max() for c in curves_a))
xb = (min(c.real.min() for c in curves_b), max(c.real.max() for c in curves_b))
print(f"  A x-range: {xa[0]:.0f} to {xa[1]:.0f}")
print(f"  B x-range: {xb[0]:.0f} to {xb[1]:.0f}")

# ── INTERPOLATE ────────────────────────────────────────────────────────────────
print(f"\nInterpolating {STEPS} frames (with phase alignment)...")
alphas  = np.linspace(0, 1, STEPS)
blended = []

for alpha in alphas:
    phi_blend = (1 - alpha) * phi_a + alpha * phi_b_aligned
    curves    = reconstruct_curves(phi_blend, N_HARMONICS, max_curves)
    rendered  = render_curves(curves)
    blended.append((rendered, alpha))
    print(f"  alpha={alpha:.2f} — {len(curves)} curves")

# ── PLOT ───────────────────────────────────────────────────────────────────────
total_cols = 2 + STEPS + 2   # origA | reconA | blends | reconB | origB

fig = plt.figure(figsize=(3 * total_cols, 5), facecolor=BG)
fig.suptitle(
    f'Heptapod B — Logogram Interpolation in \u03a6(L) Space\n'
    f'#{IDX_A} ({names[IDX_A][:16]})  \u2192  #{IDX_B} ({names[IDX_B][:16]})',
    color=WHITE, fontsize=13, fontweight='bold', y=0.99)

gs = gridspec.GridSpec(1, total_cols, figure=fig, wspace=0.04)

def show(slot, img, title, color):
    ax = fig.add_subplot(gs[0, slot])
    ax.imshow(img, cmap='gray', vmin=0, vmax=255)
    ax.set_title(title, color=color, fontsize=8, fontweight='bold', pad=5)
    ax.axis('off')
    for s in ax.spines.values():
        s.set_color(color); s.set_linewidth(2.5); s.set_visible(True)

show(0, img_a,   f'Original A\n#{IDX_A}', CYAN)
show(1, recon_a, 'Recon A \u2713',        CYAN)

for i, (rendered, alpha) in enumerate(blended):
    col = CYAN if alpha < 0.1 else (ORANGE if alpha > 0.9 else GRAY)
    show(i + 2, rendered, f'\u03b1 = {alpha:.2f}', col)

show(total_cols - 2, recon_b, 'Recon B \u2713',        ORANGE)
show(total_cols - 1, img_b,   f'Original B\n#{IDX_B}', ORANGE)

out_path = f'interpolation_{IDX_A}_to_{IDX_B}.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print(f"\nSaved: {out_path}")