"""
generate_wireframes.py
======================
Generates skeletal Fourier wireframe images for every logogram in the dataset.
Output images are named identically to the input files so the website can
load them from logograms_wire/000.png through logograms_wire/048.png.

Usage:
    python generate_wireframes.py
    python generate_wireframes.py --input logograms/ --output logograms_wire/

Output:
    logograms_wire/<same_filename>.png  for every PNG in the input folder
    One summary contact sheet: logograms_wire/_all.png
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from heptapod_encode import (
    encode_logogram, reconstruct_curves, _build_phi
)

# ── CONFIG ─────────────────────────────────────────────────────────────────────
DEFAULT_INPUT  = "logograms"
DEFAULT_OUTPUT = "logograms_wire"
IMG_SIZE       = 512      # output image size in pixels
BASE_WIDTH     = 4        # base stroke width
PRESSURE_VAR   = 0.3      # sinusoidal pressure variation amplitude
JITTER_STD     = 0.6      # positional jitter (organic line feel)
TAPER          = True     # taper strokes at endpoints
SPLATTER       = True     # add ink splatter dots at high-pressure points
N_HARMONICS    = 64


# ── BRUSHSTROKE RENDERER ───────────────────────────────────────────────────────
def render_brushstroke(curves, img_size=IMG_SIZE, base_width=BASE_WIDTH,
                        pressure_var=PRESSURE_VAR, jitter_std=JITTER_STD,
                        taper=TAPER, splatter=SPLATTER, seed=42):
    """
    Render Fourier-reconstructed curves as a variable-width brushstroke image.
    Returns a (img_size x img_size) uint8 numpy array (white background, black ink).
    """
    canvas = np.ones((img_size, img_size), dtype=np.uint8) * 255
    if not curves:
        return canvas

    # Auto-fit all curves jointly to canvas
    all_x = np.concatenate([c.real for c in curves])
    all_y = np.concatenate([c.imag for c in curves])
    x_min, x_max = all_x.min(), all_x.max()
    y_min, y_max = all_y.min(), all_y.max()
    pad_px = img_size * 0.1
    usable  = img_size - 2 * pad_px
    scale   = usable / max(x_max - x_min, y_max - y_min, 1.0)
    cx      = pad_px + (usable - (x_max - x_min) * scale) / 2
    cy      = pad_px + (usable - (y_max - y_min) * scale) / 2

    rng = np.random.default_rng(seed)

    for ci, curve in enumerate(curves):
        n  = len(curve)
        xs = ((curve.real - x_min) * scale + cx).astype(np.float32)
        ys = ((curve.imag - y_min) * scale + cy).astype(np.float32)

        # Downsample to ~400 points for speed without losing shape
        step   = max(1, n // 400)
        xs, ys = xs[::step], ys[::step]
        n_pts  = len(xs)

        for j in range(n_pts - 1):
            t = j / n_pts

            # Variable pressure — sinusoidal, offset per curve
            pressure = 1.0 + pressure_var * np.sin(t * np.pi * 4 + ci * 1.3)

            # Taper at stroke endpoints
            tap = min(1.0, t * 10, (1 - t) * 10) if taper else 1.0

            width = max(1, int(base_width * pressure * tap))

            # Positional jitter for organic line quality
            jx = rng.normal(0, jitter_std)
            jy = rng.normal(0, jitter_std)

            p1 = (int(np.clip(xs[j]   + jx, 0, img_size - 1)),
                  int(np.clip(ys[j]   + jy, 0, img_size - 1)))
            p2 = (int(np.clip(xs[j+1] + jx, 0, img_size - 1)),
                  int(np.clip(ys[j+1] + jy, 0, img_size - 1)))

            cv2.line(canvas, p1, p2, 0, width)

            # Ink splatter at high-pressure points
            if splatter and pressure > 1.2 and rng.random() < 0.08:
                sx = int(np.clip(xs[j] + rng.normal(0, 5), 0, img_size - 1))
                sy = int(np.clip(ys[j] + rng.normal(0, 5), 0, img_size - 1))
                cv2.circle(canvas, (sx, sy), int(rng.integers(1, 3)), 0, -1)

    return canvas


# ── MAIN ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Generate wireframe images for AlphaLogo website")
    parser.add_argument("--input",  default=DEFAULT_INPUT,  help="Folder of logogram PNGs")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output folder for wireframes")
    parser.add_argument("--size",   type=int, default=IMG_SIZE, help="Output image size in px")
    args = parser.parse_args()

    in_folder  = Path(args.input)
    out_folder = Path(args.output)
    out_folder.mkdir(parents=True, exist_ok=True)

    png_files = sorted(in_folder.glob("*.png"))
    if not png_files:
        print(f"No PNG files found in {in_folder}")
        sys.exit(1)

    print(f"Generating wireframes for {len(png_files)} logograms")
    print(f"  Input  : {in_folder}")
    print(f"  Output : {out_folder}")
    print(f"  Size   : {args.size}×{args.size}px")
    print()

    failed    = []
    generated = []

    for idx, png_path in enumerate(png_files):
        try:
            # Encode
            enc        = encode_logogram(str(png_path), N_harmonics=N_HARMONICS)
            max_curves = len(enc.curves)
            phi        = _build_phi(enc.curves, max_curves)

            # Reconstruct
            curves = reconstruct_curves(phi, N_HARMONICS, max_curves)
            if not curves:
                raise ValueError("No curves reconstructed")

            # Render with brushstroke
            canvas = render_brushstroke(curves, img_size=args.size, seed=idx)

            # Save with SAME filename as input
            out_path = out_folder / png_path.name
            cv2.imwrite(str(out_path), canvas)
            generated.append(out_path)

            print(f"  [{idx+1:02d}/{len(png_files)}] ✓  {png_path.name}"
                  f"  ({max_curves} curves)")

        except Exception as e:
            failed.append((png_path.name, str(e)))
            print(f"  [{idx+1:02d}/{len(png_files)}] ✗  {png_path.name}: {e}")

    # ── CONTACT SHEET ─────────────────────────────────────────────────────────
    print(f"\nBuilding contact sheet...")
    n   = len(generated)
    if n > 0:
        ncols  = 7
        nrows  = (n + ncols - 1) // ncols
        fig    = plt.figure(figsize=(ncols * 2, nrows * 2.4 + 1),
                            facecolor='#0d0d0d')
        fig.suptitle('AlphaLogo — Wireframe Reconstructions',
                     color='white', fontsize=11, fontweight='bold', y=0.99)
        gs = gridspec.GridSpec(nrows, ncols, figure=fig,
                               hspace=0.3, wspace=0.05)

        for i, out_path in enumerate(generated):
            row, col = divmod(i, ncols)
            ax = fig.add_subplot(gs[row, col])
            img = cv2.imread(str(out_path), cv2.IMREAD_GRAYSCALE)
            ax.imshow(img, cmap='gray', vmin=0, vmax=255)
            ax.set_title(out_path.stem, color='#705DAA',
                         fontsize=6, pad=2)
            ax.axis('off')

        sheet_path = out_folder / '_all.png'
        plt.savefig(str(sheet_path), dpi=100, bbox_inches='tight',
                    facecolor='#0d0d0d')
        plt.close()
        print(f"  Contact sheet: {sheet_path}")

    # ── SUMMARY ───────────────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"DONE")
    print(f"{'='*50}")
    print(f"  Generated : {len(generated)}/{len(png_files)}")
    if failed:
        print(f"  Failed    : {len(failed)}")
        for name, err in failed:
            print(f"    {name}: {err}")
    print(f"\n  Output in: {out_folder}/")
    print(f"  Place logograms_wire/ alongside logograms/ and website/")
    print(f"  for the AlphaLogo website to load skeletal views.")


if __name__ == "__main__":
    main()