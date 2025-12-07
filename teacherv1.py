# teacherv1.py
#
# Stage 1: build teacher features and Δ-PCA space from paired (real, gen) images.
# Outputs in ARTIFACTS_DIR:
#   - teacher_features.npz: fr_arr, fg_arr, pair_ids, real_paths, gen_paths
#   - teacher_scaler_delta.pkl: StandardScaler fitted on Δ
#   - teacher_pca.pkl: PCA fitted on Δ (teacher latent space)

import os
import pathlib
from typing import List, Tuple

import numpy as np
import cv2

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib  # pip install joblib

# ============================
#           CONFIG
# ============================

# Folders that contain paired images:
#   real: name.jpg / name.jpeg / name.png
#   fake: name_gen.jpg / name_gen.jpeg / name_gen.png
DATA_DIRS = [
    pathlib.Path("d3/batch1_all"),
    pathlib.Path("d3/batch2_all"),
    pathlib.Path("d3/batch3_all"),
]

GEN_SUFFIX = "_gen"
IMG_EXTS = {".jpg", ".jpeg", ".png"}

# Teacher feature preprocessing:
TARGET_WIDTH  = 1280
TARGET_HEIGHT = 840
JPEG_QUALITY  = 92      # None to disable JPEG round-trip
BLUR_SIGMA    = 0.0     # 0 to disable micro-blur

# Feature toggles: same as stable v2 (no wavelet/DCT yet)
USE_RADIAL_FFT     = True
USE_LAPLACIAN_MS   = True
USE_PATCH_CONTRAST = True
USE_WAVELET_ENERGY = False
USE_DCT_BANDS      = False

# Feature params
N_RINGS    = 36
LAP_SIGMAS = [0.8, 1.6, 3.0]
PATCH_GRID = 5

# Teacher PCA space
PCA_COMPONENTS = 8

# Where to save teacher artifacts
ARTIFACTS_DIR = pathlib.Path("artifacts_ts1")

# ============================
#     PREPROCESSING HELPERS
# ============================

def imread_rgb_float01(path: pathlib.Path) -> np.ndarray:
    """Read image with OpenCV -> RGB float32 [0,1]."""
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Failed to read image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return rgb

def jpeg_roundtrip(img_rgb01: np.ndarray, quality: int) -> np.ndarray:
    """In-memory JPEG encode/decode to normalize compression."""
    arr8 = np.clip(img_rgb01 * 255.0, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(arr8, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return img_rgb01
    bgr2 = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    rgb2 = cv2.cvtColor(bgr2, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return rgb2

def preprocess_fixed_rgb01(path: pathlib.Path) -> np.ndarray:
    """
    Shared preprocessing for teacher & later student:
      - Read RGB [0,1]
      - Require at least TARGET_WIDTH x TARGET_HEIGHT
      - Center-crop to exactly (TARGET_HEIGHT, TARGET_WIDTH, 3)
      - Optional JPEG roundtrip
      - Optional Gaussian blur
    Returns: float32 [H,W,3] in [0,1], or raises if too small.
    """
    img = imread_rgb_float01(path)
    h, w = img.shape[:2]

    if w < TARGET_WIDTH or h < TARGET_HEIGHT:
        raise RuntimeError(
            f"Image too small for {TARGET_WIDTH}x{TARGET_HEIGHT} crop: {path.name} ({w}x{h})"
        )

    # Center-crop to target size
    y0 = (h - TARGET_HEIGHT) // 2
    x0 = (w - TARGET_WIDTH) // 2
    img = img[y0:y0 + TARGET_HEIGHT, x0:x0 + TARGET_WIDTH, :]

    if JPEG_QUALITY is not None:
        img = jpeg_roundtrip(img, JPEG_QUALITY)

    if BLUR_SIGMA and BLUR_SIGMA > 0:
        arr8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        arr8 = cv2.GaussianBlur(arr8, (0, 0), BLUR_SIGMA)
        img = arr8.astype(np.float32) / 255.0

    return img

# ============================
#    FEATURE EXTRACTORS
# ============================

def f_radial_fft(img: np.ndarray, n_rings: int = N_RINGS) -> np.ndarray:
    """Radial FFT energy: mean magnitude in concentric rings."""
    gray = cv2.cvtColor((img * 255.0).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    F = np.fft.fftshift(np.fft.fft2(gray))
    mag = np.abs(F)
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.indices((h, w))
    rr = np.sqrt((xx - cx)**2 + (yy - cy)**2)
    r_norm = rr / (rr.max() + 1e-12)
    edges = np.linspace(0.0, 1.0, n_rings + 1)
    feats = []
    for i in range(n_rings):
        m = (r_norm >= edges[i]) & (r_norm < edges[i+1])
        feats.append(float(mag[m].mean()) if np.any(m) else 0.0)
    return np.array(feats, dtype=np.float32)

def f_laplacian_ms(img: np.ndarray, sigmas: List[float] = LAP_SIGMAS) -> np.ndarray:
    """Multi-scale Laplacian variance."""
    gray = cv2.cvtColor((img * 255.0).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    vals = []
    for s in sigmas:
        blur = cv2.GaussianBlur(gray, (0, 0), s)
        lap  = cv2.Laplacian(blur, cv2.CV_32F)
        vals.append(float(lap.var()))
    return np.array(vals, dtype=np.float32)

def f_patch_contrast(img: np.ndarray, grid: int = PATCH_GRID) -> np.ndarray:
    """Patch contrast: stddev per patch in a grid."""
    gray = cv2.cvtColor((img * 255.0).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    h, w = gray.shape
    ph, pw = max(1, h // grid), max(1, w // grid)
    vals = []
    for i in range(grid):
        for j in range(grid):
            p = gray[i*ph:min((i+1)*ph, h), j*pw:min((j+1)*pw, w)]
            vals.append(float(p.std()))
    return np.array(vals, dtype=np.float32)

def extract_features_single(img: np.ndarray) -> np.ndarray:
    feats = []
    if USE_RADIAL_FFT:
        feats.append(f_radial_fft(img))
    if USE_LAPLACIAN_MS:
        feats.append(f_laplacian_ms(img))
    if USE_PATCH_CONTRAST:
        feats.append(f_patch_contrast(img))
    # wavelet/DCT off for now
    if not feats:
        raise RuntimeError("No feature families enabled.")
    return np.concatenate(feats, axis=0).astype(np.float32)

# ============================
#       PAIR DISCOVERY
# ============================

def list_pairs_one_dir(root: pathlib.Path) -> Tuple[List[Tuple[pathlib.Path, pathlib.Path, str]], List[pathlib.Path]]:
    """
    Find (real, gen) pairs in a directory:
      real: stem = name
      gen:  stem = name_gen
    Returns:
      pairs: [(real_path, gen_path, pair_id), ...]
      missing: [real_paths with no gen]
    """
    files = [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    name_map = {p.name.lower(): p for p in files}
    reals = [p for p in files if not p.stem.lower().endswith(GEN_SUFFIX)]

    pairs = []
    missing = []
    for r in sorted(reals, key=lambda x: x.name.lower()):
        base = r.stem
        found = None
        for ext in IMG_EXTS:
            cand = f"{base}{GEN_SUFFIX}{ext}".lower()
            if cand in name_map:
                found = name_map[cand]
                break
        if found is None:
            missing.append(r)
        else:
            pair_id = f"{root.name}/{base}"
            pairs.append((r, found, pair_id))
    return pairs, missing

def collect_pairs(data_dirs: List[pathlib.Path]):
    all_pairs = []
    all_missing = []
    for d in data_dirs:
        if not d.exists():
            print(f"[warn] DATA_DIR missing: {d}")
            continue
        p, m = list_pairs_one_dir(d)
        all_pairs.extend(p)
        all_missing.extend(m)
    return all_pairs, all_missing

# ============================
#           MAIN
# ============================

def main():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    pairs, missing = collect_pairs(DATA_DIRS)
    print(f"[pairing] total pairs (before size filter): {len(pairs)}; missing-gen for {len(missing)} reals")
    if missing[:5]:
        print("[pairing] first few missing:", [m.name for m in missing[:5]])

    fr_list, fg_list = [], []
    real_paths, gen_paths, pair_ids = [], [], []
    dropped_small = 0

    print(f"[teacher] extracting base features with target crop {TARGET_WIDTH}x{TARGET_HEIGHT}...")
    for idx, (rp, gp, pid) in enumerate(pairs, 1):
        try:
            r_img = preprocess_fixed_rgb01(rp)
            g_img = preprocess_fixed_rgb01(gp)
        except RuntimeError as e:
            dropped_small += 1
            if dropped_small <= 5:
                print("[drop-small]", e)
            continue

        fr = extract_features_single(r_img)
        fg = extract_features_single(g_img)

        fr_list.append(fr)
        fg_list.append(fg)
        real_paths.append(str(rp))
        gen_paths.append(str(gp))
        pair_ids.append(pid)

        if idx <= 3:
            print(f"  pair {idx}: {rp.name} vs {gp.name}, feat_dim={len(fr)}")

    if not fr_list:
        raise SystemExit("No valid pairs after size filtering; aborting.")

    fr_arr = np.vstack(fr_list)
    fg_arr = np.vstack(fg_list)
    pair_ids = np.array(pair_ids)
    real_paths = np.array(real_paths)
    gen_paths = np.array(gen_paths)

    print(f"[teacher] kept pairs: {fr_arr.shape[0]}, dropped_too_small: {dropped_small}")
    print(f"[teacher] feature dim per image: {fr_arr.shape[1]}")

    # Build Δ and fit scaler + PCA
    delta = fg_arr - fr_arr
    print(f"[teacher] Δ matrix shape: {delta.shape}")

    scaler_delta = StandardScaler().fit(delta)
    delta_scaled = scaler_delta.transform(delta)

    pca = PCA(n_components=PCA_COMPONENTS, random_state=12345).fit(delta_scaled)
    print(f"[teacher] PCA on Δ: k={PCA_COMPONENTS}, explained_var_sum={pca.explained_variance_ratio_.sum():.3f}")

    # Save artifacts
    np.savez(
        ARTIFACTS_DIR / "teacher_features.npz",
        fr_arr=fr_arr,
        fg_arr=fg_arr,
        pair_ids=pair_ids,
        real_paths=real_paths,
        gen_paths=gen_paths,
    )
    joblib.dump(scaler_delta, ARTIFACTS_DIR / "teacher_scaler_delta.pkl")
    joblib.dump(pca,          ARTIFACTS_DIR / "teacher_pca.pkl")

    print(f"[save] wrote {ARTIFACTS_DIR / 'teacher_features.npz'}")
    print(f"[save] wrote {ARTIFACTS_DIR / 'teacher_scaler_delta.pkl'}")
    print(f"[save] wrote {ARTIFACTS_DIR / 'teacher_pca.pkl'}")

if __name__ == "__main__":
    main()
