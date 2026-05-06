"""
track1_grid.py
==============
Track 1 — Full interpolation grid across all 49 logograms.

What this script does:
    1. Encodes all 49 logograms into Phi(L) space
    2. Runs interpolation between a curated set of pairs
    3. Applies geodesic normalization to fix midpoint energy drop
    4. Scores each interpolation by smoothness
    5. Saves the best pairs as paper figures
    6. Saves a summary grid of all tested pairs

Run:
    python track1_grid.py

Outputs:
    grid_summary.png          — all pairs in one overview image
    best_pair_<A>_<B>.png     — top 5 best interpolations (paper figures)
    interpolation_scores.txt  — smoothness scores for all pairs
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2
from pathlib import Path
from itertools import combinations
from heptapod_encode import (
    batch_encode, save_encodings,
    reconstruct_curves, render_curves, align_phases,
)

# ── CONFIG ─────────────────────────────────────────────────────────────────────
DATASET_FOLDER = "logograms"       # <- your 49 logograms folder
PHI_MATRIX     = "phi_matrix.npy"
N_HARMONICS    = 64
STEPS          = 7                 # frames per interpolation (odd number)
N_TOP_PAIRS    = 5                 # how many best pairs to save as paper figures

# Curated pairs to always include (add/change indices as you like)
# These are pairs that looked visually interesting from earlier experiments
CURATED_PAIRS  = [
    (0,  10),
    (10, 20),
    (10, 30),
    (16, 25),
    (16, 30),
]

# How many random additional pairs to test (set 0 to only use curated)
N_RANDOM_PAIRS = 20

BG     = '#0d0d0d'
CYAN   = '#00d4ff'
ORANGE = '#ff6b35'
WHITE  = '#f0f0f0'
GRAY   = '#666666'

# ── GEODESIC NORMALIZATION ─────────────────────────────────────────────────────
def geodesic_blend(phi_a, phi_b_aligned, alpha,
                   N_harmonics, max_curves):
    """
    Blend phi_a and phi_b with energy normalization at each step.

    Standard linear blend: phi = (1-a)*A + a*B
    Problem: midpoint energy drops because harmonics partially cancel.

    Fix: after blending, rescale the coefficient magnitudes so the total
    spectral energy matches the linear interpolation of A and B's energies.
    This keeps visual weight consistent across all frames.
    """
    phi_blend = (1 - alpha) * phi_a + alpha * phi_b_aligned

    n_coeffs    = 2 * N_harmonics + 1
    coeff_block = max_curves * n_coeffs

    # Energy of A, B, and blend
    energy_a     = np.sum(phi_a[:2*coeff_block]**2)
    energy_b     = np.sum(phi_b_aligned[:2*coeff_block]**2)
    energy_blend = np.sum(phi_blend[:2*coeff_block]**2)

    target_energy = (1 - alpha) * energy_a + alpha * energy_b

    if energy_blend > 1e-10:
        scale = np.sqrt(target_energy / energy_blend)
        phi_blend[:2*coeff_block] *= scale

    return phi_blend


# ── SMOOTHNESS SCORE ───────────────────────────────────────────────────────────
def smoothness_score(frames):
    """
    Score an interpolation sequence by how smoothly frames change.

    Method: compute the mean pixel difference between consecutive frames.
    Lower variance in frame-to-frame difference = smoother morph.
    Score is in [0, 1] where 1 = perfectly smooth.
    """
    diffs = []
    for i in range(len(frames) - 1):
        diff = np.abs(frames[i].astype(float) - frames[i+1].astype(float))
        diffs.append(diff.mean())
    diffs = np.array(diffs)
    # Smoothness = 1 - (normalized std of diffs)
    if diffs.mean() < 1e-6:
        return 0.0
    score = 1.0 - (diffs.std() / (diffs.mean() + 1e-6))
    return float(np.clip(score, 0, 1))


# ── ENCODE ─────────────────────────────────────────────────────────────────────
print("=" * 60)
print("TRACK 1 — Full Interpolation Grid")
print("=" * 60)
print(f"\nEncoding logograms from: {DATASET_FOLDER}")

encodings     = batch_encode(DATASET_FOLDER, N_harmonics=N_HARMONICS)
matrix, names = save_encodings(encodings, PHI_MATRIX)
max_curves    = max(len(encodings[n].curves) for n in names)
N             = len(names)

print(f"\nDataset   : {N} logograms")
print(f"Matrix    : {matrix.shape}")
print(f"max_curves: {max_curves}")

# ── BUILD PAIR LIST ────────────────────────────────────────────────────────────
all_pairs = list(CURATED_PAIRS)

# Add random pairs
np.random.seed(42)
available = [(i, j) for i, j in combinations(range(N), 2)
             if (i, j) not in all_pairs]
np.random.shuffle(available)
all_pairs += available[:N_RANDOM_PAIRS]

# Deduplicate and validate
seen = set()
valid_pairs = []
for a, b in all_pairs:
    if a >= N or b >= N:
        print(f"  Skipping ({a},{b}) — index out of range")
        continue
    key = (min(a,b), max(a,b))
    if key not in seen:
        seen.add(key)
        valid_pairs.append((a, b))

print(f"\nTesting {len(valid_pairs)} pairs...")

# ── RUN INTERPOLATIONS ─────────────────────────────────────────────────────────
results = []   # list of dicts

for pair_idx, (idx_a, idx_b) in enumerate(valid_pairs):
    phi_a         = matrix[idx_a]
    phi_b         = matrix[idx_b]
    phi_b_aligned = align_phases(phi_a, phi_b, N_HARMONICS, max_curves)

    alphas = np.linspace(0, 1, STEPS)
    frames = []

    for alpha in alphas:
        phi_blend = geodesic_blend(phi_a, phi_b_aligned, alpha,
                                   N_HARMONICS, max_curves)
        curves    = reconstruct_curves(phi_blend, N_HARMONICS, max_curves)
        rendered  = render_curves(curves)
        frames.append(rendered)

    score = smoothness_score(frames)

    results.append({
        'idx_a':  idx_a,
        'idx_b':  idx_b,
        'name_a': names[idx_a],
        'name_b': names[idx_b],
        'frames': frames,
        'alphas': alphas,
        'score':  score,
        'phi_a':  phi_a,
        'phi_b_aligned': phi_b_aligned,
    })

    print(f"  [{pair_idx+1:02d}/{len(valid_pairs)}] "
          f"{names[idx_a][:8]} → {names[idx_b][:8]}  "
          f"score={score:.4f}")

# Sort by score descending
results.sort(key=lambda r: r['score'], reverse=True)

# ── SAVE SCORES ───────────────────────────────────────────────────────────────
with open('interpolation_scores.txt', 'w') as f:
    f.write("Rank  Idx_A  Idx_B  Name_A              Name_B              Score\n")
    f.write("-" * 75 + "\n")
    for rank, r in enumerate(results, 1):
        f.write(f"{rank:4d}  {r['idx_a']:5d}  {r['idx_b']:5d}  "
                f"{r['name_a'][:18]:<18}  {r['name_b'][:18]:<18}  "
                f"{r['score']:.4f}\n")

print(f"\nScores saved → interpolation_scores.txt")
print(f"\nTop 5 pairs:")
for r in results[:5]:
    print(f"  {r['name_a'][:12]} → {r['name_b'][:12]}  score={r['score']:.4f}")

# ── SAVE BEST PAIRS AS PAPER FIGURES ─────────────────────────────────────────
print(f"\nSaving top {N_TOP_PAIRS} pairs as paper figures...")

for rank, r in enumerate(results[:N_TOP_PAIRS], 1):
    img_a = cv2.imread(str(Path(DATASET_FOLDER) / r['name_a']), cv2.IMREAD_GRAYSCALE)
    img_b = cv2.imread(str(Path(DATASET_FOLDER) / r['name_b']), cv2.IMREAD_GRAYSCALE)

    # Reconstruct originals for comparison
    curves_a = reconstruct_curves(r['phi_a'],         N_HARMONICS, max_curves)
    curves_b = reconstruct_curves(r['phi_b_aligned'], N_HARMONICS, max_curves)
    recon_a  = render_curves(curves_a)
    recon_b  = render_curves(curves_b)

    total = 2 + STEPS + 2
    fig   = plt.figure(figsize=(3*total, 5), facecolor=BG)
    fig.suptitle(
        f'Heptapod B — Φ(L) Interpolation  [Rank #{rank}, score={r["score"]:.4f}]\n'
        f'#{r["idx_a"]} ({r["name_a"][:14]})  →  #{r["idx_b"]} ({r["name_b"][:14]})',
        color=WHITE, fontsize=12, fontweight='bold', y=0.99)

    gs = gridspec.GridSpec(1, total, figure=fig, wspace=0.04)

    def show(slot, img, title, color):
        ax = fig.add_subplot(gs[0, slot])
        ax.imshow(img, cmap='gray', vmin=0, vmax=255)
        ax.set_title(title, color=color, fontsize=8, fontweight='bold', pad=4)
        ax.axis('off')
        for s in ax.spines.values():
            s.set_color(color); s.set_linewidth(2); s.set_visible(True)

    show(0, img_a,   f'Original A\n#{r["idx_a"]}', CYAN)
    show(1, recon_a, 'Recon A ✓',                  CYAN)
    for i, (frame, alpha) in enumerate(zip(r['frames'], r['alphas'])):
        col = CYAN if alpha < 0.1 else (ORANGE if alpha > 0.9 else GRAY)
        show(i+2, frame, f'α={alpha:.2f}', col)
    show(total-2, recon_b, 'Recon B ✓',                  ORANGE)
    show(total-1, img_b,   f'Original B\n#{r["idx_b"]}', ORANGE)

    fname = f'best_pair_{r["idx_a"]}_{r["idx_b"]}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"  Saved: {fname}")

# ── GRID SUMMARY ──────────────────────────────────────────────────────────────
print(f"\nBuilding summary grid...")

# Show top 15 pairs in a compact grid (just the 7 frames, no originals)
N_GRID   = min(15, len(results))
fig_grid = plt.figure(figsize=(STEPS*2, N_GRID*2.2), facecolor=BG)
fig_grid.suptitle('Heptapod B — Interpolation Grid Summary\n'
                  'Ordered by smoothness score (best at top)',
                  color=WHITE, fontsize=13, fontweight='bold', y=1.01)

gs_grid = gridspec.GridSpec(N_GRID, STEPS + 2,
                             figure=fig_grid,
                             hspace=0.05, wspace=0.03)

for row, r in enumerate(results[:N_GRID]):
    # Label column
    ax_label = fig_grid.add_subplot(gs_grid[row, 0])
    ax_label.set_facecolor(BG)
    ax_label.axis('off')
    ax_label.text(0.5, 0.5,
                  f"#{r['idx_a']}→#{r['idx_b']}\n{r['score']:.3f}",
                  color=CYAN, fontsize=7, ha='center', va='center',
                  fontfamily='monospace')

    for col, (frame, alpha) in enumerate(zip(r['frames'], r['alphas'])):
        ax = fig_grid.add_subplot(gs_grid[row, col+1])
        ax.imshow(frame, cmap='gray', vmin=0, vmax=255)
        if row == 0:
            ax.set_title(f'α={alpha:.2f}', color=WHITE, fontsize=7, pad=2)
        ax.axis('off')

    # Score bar on right
    ax_score = fig_grid.add_subplot(gs_grid[row, -1])
    ax_score.barh([0], [r['score']], color=CYAN, alpha=0.7)
    ax_score.set_xlim(0, 1)
    ax_score.set_facecolor(BG)
    ax_score.axis('off')
    ax_score.text(r['score']+0.02, 0, f"{r['score']:.3f}",
                  color=WHITE, fontsize=6, va='center')

plt.savefig('grid_summary.png', dpi=120, bbox_inches='tight', facecolor=BG)
plt.close()
print("Saved: grid_summary.png")

print("\n" + "="*60)
print("TRACK 1 COMPLETE")
print("="*60)
print(f"  Total pairs tested : {len(results)}")
print(f"  Best score         : {results[0]['score']:.4f}")
print(f"  Worst score        : {results[-1]['score']:.4f}")
print(f"  Mean score         : {np.mean([r['score'] for r in results]):.4f}")
print(f"\nFiles saved:")
print(f"  interpolation_scores.txt")
print(f"  grid_summary.png")
print(f"  best_pair_*.png  ({N_TOP_PAIRS} files)")
print(f"\nSend the grid_summary.png and the scores file back")
print(f"and we will write the paper.")