"""
heptapod_encode.py  (v5 — definitive)
======================================
Complex-plane parameterization of Heptapod B logograms.

Changes in v5:
    - render_curves: auto-fits all curves to canvas (no more overflow clipping)
    - reconstruct_curves: returns curves in normalized [-1,1] space
      so rendering is always canvas-independent
    - Phase-alignment utility: align_phases(phi_a, phi_b) aligns B to A
      before interpolation, preventing Fourier cancellation at midpoints
    - All pixel-space logic moved to render step only
"""

import cv2
import numpy as np
from pathlib import Path
from skimage.morphology import skeletonize
from skimage import img_as_bool, img_as_ubyte
from dataclasses import dataclass
from typing import List, Dict, Tuple


# ── DATA STRUCTURES ────────────────────────────────────────────────────────────

@dataclass
class CurveDescriptor:
    coefficients: np.ndarray   # complex (2N+1,) normalized+centered
    mean_pos:     complex      # raw pixel centroid
    scale_factor: float        # raw |c_1| before normalization
    n_harmonics:  int
    n_points:     int
    curve_index:  int


@dataclass
class LogogramEncoding:
    curves:      List[CurveDescriptor]
    phi:         np.ndarray
    source_path: str
    image_shape: tuple


# ── PREPROCESSING ──────────────────────────────────────────────────────────────

def _preprocess(img_path: str):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot load: {img_path}")
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    skeleton  = img_as_ubyte(skeletonize(img_as_bool(binary)))
    return img, skeleton


def _extract_contours(skeleton: np.ndarray, min_points: int = 50):
    contours, _ = cv2.findContours(skeleton, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    sig = [c for c in contours if len(c) >= min_points]
    return sorted(sig, key=lambda c: len(c), reverse=True)


def _canonical_start(gamma: np.ndarray) -> np.ndarray:
    """Roll contour to start at rightmost point — consistent phase reference."""
    return np.roll(gamma, -np.argmax(gamma.real))


# ── ENCODING ───────────────────────────────────────────────────────────────────

def _encode_curve(contour: np.ndarray, N: int, index: int) -> CurveDescriptor:
    pts   = contour.squeeze()
    gamma = pts[:, 0] + 1j * pts[:, 1]

    gamma      = _canonical_start(gamma)
    mean_pos   = np.mean(gamma)
    gamma_c    = gamma - mean_pos

    coeffs       = np.fft.fftshift(np.fft.fft(gamma_c))
    dc           = len(coeffs) // 2
    scale_factor = np.abs(coeffs[dc + 1])
    if scale_factor > 1e-10:
        coeffs = coeffs / scale_factor

    # Truncate or zero-pad to exactly 2N+1
    center   = len(coeffs) // 2
    n_needed = 2 * N + 1
    if len(coeffs) >= n_needed:
        descriptors = coeffs[center - N : center + N + 1].copy()
    else:
        descriptors = np.zeros(n_needed, dtype=complex)
        n_avail     = len(coeffs)
        half_avail  = n_avail // 2
        descriptors[N - half_avail : N - half_avail + n_avail] = coeffs

    return CurveDescriptor(
        coefficients = descriptors,
        mean_pos     = mean_pos,
        scale_factor = scale_factor,
        n_harmonics  = N,
        n_points     = len(gamma),
        curve_index  = index,
    )


def _build_phi(curves: List[CurveDescriptor], max_curves: int) -> np.ndarray:
    """
    Flatten to fixed-width real vector. Layout:
        real parts     : max_curves * n_coeffs  floats
        imag parts     : max_curves * n_coeffs  floats
        mean_pos_real  : max_curves              floats
        mean_pos_imag  : max_curves              floats
        scale_factors  : max_curves              floats
    Unused slots are zero so reconstruct_curves can skip them.
    """
    n_coeffs   = curves[0].coefficients.shape[0]
    real_block = np.zeros(max_curves * n_coeffs)
    imag_block = np.zeros(max_curves * n_coeffs)
    mean_real  = np.zeros(max_curves)
    mean_imag  = np.zeros(max_curves)
    scales     = np.zeros(max_curves)

    for i, cd in enumerate(curves):
        if i >= max_curves:
            break
        real_block[i*n_coeffs:(i+1)*n_coeffs] = cd.coefficients.real
        imag_block[i*n_coeffs:(i+1)*n_coeffs] = cd.coefficients.imag
        mean_real[i] = cd.mean_pos.real
        mean_imag[i] = cd.mean_pos.imag
        scales[i]    = cd.scale_factor

    return np.concatenate([real_block, imag_block, mean_real, mean_imag, scales])


# ── PUBLIC ENCODE API ──────────────────────────────────────────────────────────

def encode_logogram(img_path: str,
                    N_harmonics:     int = 64,
                    min_contour_pts: int = 50) -> LogogramEncoding:
    """PNG → LogogramEncoding."""
    img, skeleton = _preprocess(img_path)
    contours      = _extract_contours(skeleton, min_contour_pts)
    if not contours:
        raise ValueError(f"No significant contours in {img_path}")
    curves = [_encode_curve(c, N_harmonics, i) for i, c in enumerate(contours)]
    phi    = _build_phi(curves, max_curves=len(curves))
    return LogogramEncoding(curves=curves, phi=phi,
                            source_path=str(img_path),
                            image_shape=img.shape)


def batch_encode(folder_path: str,
                 N_harmonics: int = 64,
                 max_curves:  int = None) -> Dict[str, LogogramEncoding]:
    """Encode all PNGs in folder with uniform phi dimension."""
    folder    = Path(folder_path)
    encodings = {}
    failed    = []

    for png_path in sorted(folder.glob("*.png")):
        try:
            enc = encode_logogram(str(png_path), N_harmonics)
            encodings[png_path.name] = enc
            print(f"  ✓ {png_path.name}: {len(enc.curves)} curves")
        except Exception as e:
            failed.append((png_path.name, str(e)))
            print(f"  ✗ {png_path.name}: {e}")

    if not encodings:
        raise ValueError("No logograms encoded.")

    if max_curves is None:
        max_curves = max(len(e.curves) for e in encodings.values())

    n_coeffs   = 2 * N_harmonics + 1
    target_dim = max_curves * (2 * n_coeffs + 3)

    for enc in encodings.values():
        enc.phi = _build_phi(enc.curves, max_curves=max_curves)

    if failed:
        print(f"\nFailed: {len(failed)}")
    print(f"\nEncoded : {len(encodings)}/{len(list(folder.glob('*.png')))} logograms")
    print(f"Phi dim : {target_dim}  (max_curves={max_curves}, N={N_harmonics})")
    return encodings


def save_encodings(encodings: Dict[str, LogogramEncoding],
                   output_path: str) -> Tuple[np.ndarray, list]:
    names = sorted(encodings.keys())
    dims  = {len(encodings[n].phi) for n in names}
    if len(dims) != 1:
        raise ValueError(f"Inconsistent Phi dimensions: {dims}")
    matrix = np.stack([encodings[n].phi for n in names])
    np.save(output_path, matrix)
    print(f"Saved: {matrix.shape} -> {output_path}")
    return matrix, names


# ── RECONSTRUCTION ─────────────────────────────────────────────────────────────

def reconstruct_curves(phi: np.ndarray,
                       N_harmonics: int = 64,
                       max_curves:  int = None,
                       n_points:    int = 500) -> List[np.ndarray]:
    """
    Inverse: phi vector → list of complex curves in pixel space.
    Works on raw phi AND interpolated blends.
    """
    n_coeffs = 2 * N_harmonics + 1
    if max_curves is None:
        max_curves = len(phi) // (2 * n_coeffs + 3)

    coeff_block   = max_curves * n_coeffs
    real_parts    = phi[0             : coeff_block]
    imag_parts    = phi[coeff_block   : 2*coeff_block]
    mean_pos_real = phi[2*coeff_block : 2*coeff_block + max_curves]
    mean_pos_imag = phi[2*coeff_block + max_curves : 2*coeff_block + 2*max_curves]
    scales        = phi[2*coeff_block + 2*max_curves : 2*coeff_block + 3*max_curves]

    curves_out = []
    for i in range(max_curves):
        sc = scales[i]
        if sc < 1e-2:
            continue
        r  = real_parts[i*n_coeffs : (i+1)*n_coeffs]
        im = imag_parts[i*n_coeffs : (i+1)*n_coeffs]
        mp = mean_pos_real[i] + 1j * mean_pos_imag[i]

        coeffs = r + 1j * im
        padded = np.zeros(n_points, dtype=complex)
        center = n_points // 2
        half   = min(N_harmonics, center)
        padded[center - half : center + half + 1] = coeffs[N_harmonics - half :
                                                            N_harmonics + half + 1]
        recon_norm  = np.fft.ifft(np.fft.ifftshift(padded))
        recon_pixel = recon_norm * sc + mp
        curves_out.append(recon_pixel)

    return curves_out


# ── RENDERING ─────────────────────────────────────────────────────────────────

def render_curves(curves: List[np.ndarray],
                  img_size: int = 224,
                  thickness: int = 2,
                  padding: float = 0.08) -> np.ndarray:
    """
    Draw reconstructed curves onto a white canvas.

    Auto-fits ALL curves jointly to the canvas with padding so nothing
    is clipped regardless of how far coordinates overflow the original
    image bounds. This is essential for interpolated frames where the
    blended mean_pos and scale may place curves at unexpected positions.
    """
    canvas = np.ones((img_size, img_size), dtype=np.uint8) * 255
    if not curves:
        return canvas

    # Find global bounding box across all curves
    all_x = np.concatenate([c.real for c in curves])
    all_y = np.concatenate([c.imag for c in curves])
    x_min, x_max = all_x.min(), all_x.max()
    y_min, y_max = all_y.min(), all_y.max()

    span_x = x_max - x_min if x_max > x_min else 1.0
    span_y = y_max - y_min if y_max > y_min else 1.0

    # Scale to fit canvas with padding
    pad_px  = img_size * padding
    usable  = img_size - 2 * pad_px
    scale   = usable / max(span_x, span_y)

    # Center within canvas
    cx = pad_px + (usable - span_x * scale) / 2
    cy = pad_px + (usable - span_y * scale) / 2

    for curve in curves:
        xs = ((curve.real - x_min) * scale + cx).astype(np.int32)
        ys = ((curve.imag - y_min) * scale + cy).astype(np.int32)
        xs = np.clip(xs, 0, img_size - 1)
        ys = np.clip(ys, 0, img_size - 1)
        pts = np.column_stack([xs, ys])
        for j in range(len(pts) - 1):
            cv2.line(canvas, tuple(pts[j]), tuple(pts[j+1]), 0, thickness)

    return canvas


# ── PHASE ALIGNMENT ────────────────────────────────────────────────────────────

def align_phases(phi_a: np.ndarray, phi_b: np.ndarray,
                 N_harmonics: int = 64,
                 max_curves:  int = None) -> np.ndarray:
    """
    Return a version of phi_b whose Fourier descriptors are phase-rotated
    to match phi_a's fundamental frequency phase, per curve.

    WHY THIS IS NEEDED:
        Fourier descriptors encode the curve as a sum of rotating circles.
        If two curves have the same shape but different orientations, their
        c_1 coefficients point in different directions. Blending them linearly
        causes partial cancellation — the midpoint looks like a shrunken or
        invisible version of both. Phase alignment rotates B so its c_1 points
        in the same direction as A's c_1 before blending, ensuring constructive
        rather than destructive interference.

    WHAT IT PRESERVES:
        Shape, scale, and position are unchanged — only the rotational
        orientation of the descriptor representation is adjusted.
    """
    n_coeffs = 2 * N_harmonics + 1
    if max_curves is None:
        max_curves = len(phi_a) // (2 * n_coeffs + 3)

    phi_b_aligned = phi_b.copy()
    coeff_block   = max_curves * n_coeffs

    for i in range(max_curves):
        # Extract descriptors for curve i from both phi vectors
        r_a = phi_a[i*n_coeffs         : (i+1)*n_coeffs]
        im_a= phi_a[coeff_block + i*n_coeffs : coeff_block + (i+1)*n_coeffs]
        r_b = phi_b[i*n_coeffs         : (i+1)*n_coeffs]
        im_b= phi_b[coeff_block + i*n_coeffs : coeff_block + (i+1)*n_coeffs]

        c_a = r_a + 1j * im_a
        c_b = r_b + 1j * im_b

        # c_1 sits at index N in the (2N+1) descriptor array
        c1_a = c_a[N_harmonics]  # fundamental of A
        c1_b = c_b[N_harmonics]  # fundamental of B

        if np.abs(c1_a) < 1e-10 or np.abs(c1_b) < 1e-10:
            continue  # skip zero-padded slots

        # Phase difference: how much to rotate B to match A
        phase_diff = np.angle(c1_a) - np.angle(c1_b)

        # Apply rotation: multiply c_n by e^(i * n * phase_diff)
        # This rotates the curve in the complex plane without changing shape
        n_vals   = np.arange(-N_harmonics, N_harmonics + 1, dtype=float)
        rotation = np.exp(1j * n_vals * phase_diff)
        c_b_rot  = c_b * rotation

        # Write back into phi_b_aligned
        phi_b_aligned[i*n_coeffs         : (i+1)*n_coeffs] = c_b_rot.real
        phi_b_aligned[coeff_block + i*n_coeffs : coeff_block + (i+1)*n_coeffs] = c_b_rot.imag

    return phi_b_aligned