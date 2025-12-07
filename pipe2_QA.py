# pipe2_QA.py
# score_and_filter_pipe1.py
import shutil, pathlib, numpy as np, cv2, torch
from PIL import Image, ImageOps
from skimage.metrics import structural_similarity as ssim
import lpips

# ------------ CONFIG ------------
REF_DIR   = pathlib.Path("ref_new/batch1")
GEN_DIR   = pathlib.Path("ref_new/batch1_gen")
FINAL_DIR = pathlib.Path("ref_new/batch1_final")

MAX_WORK_SIDE = 1280      # must match your generator
ROUND_TO_8    = True
SAT_MATCH     = False     # set True to reduce vibrancy mismatch

THRESHOLD_SSIM  = 0.80
THRESHOLD_LPIPS = 0.20
# --------------------------------


def round_to_8(x): return int(8 * round(float(x)/8))

def load_rgb_pil(path):
    im = Image.open(path)
    im = ImageOps.exif_transpose(im)    # fix rotated EXIF images
    return im.convert("RGB")

def resize_full_like_gen(im_pil):
    w, h = im_pil.size
    s = max(w, h)
    if s > MAX_WORK_SIDE:
        scale = MAX_WORK_SIDE / s
        im_pil = im_pil.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
    if ROUND_TO_8:
        w, h = im_pil.size
        im_pil = im_pil.resize((round_to_8(w), round_to_8(h)), Image.LANCZOS)
    return im_pil

def to_np(im_pil):
    return np.asarray(im_pil)

def match_sat(ref, gen, cap=1.25):
    def to_hsv(img):
        return cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    rh, gh = to_hsv(ref), to_hsv(gen)
    Sr, Sg = rh[...,1].mean(), gh[...,1].mean()
    if Sr > 0 and Sg > 0:
        if Sr > Sg:
            scale = max(Sg / Sr, 1.0/cap)
            rh[...,1] = np.clip(rh[...,1] * scale, 0, 255)
        else:
            scale = max(Sr / Sg, 1.0/cap)
            gh[...,1] = np.clip(gh[...,1] * scale, 0, 255)
    return (
        cv2.cvtColor(rh.astype(np.uint8), cv2.COLOR_HSV2RGB),
        cv2.cvtColor(gh.astype(np.uint8), cv2.COLOR_HSV2RGB),
    )

def to_lpips_tensor(img_np, device):
    t = torch.from_numpy(img_np).permute(2,0,1).float()[None] / 255.0
    return (t*2 - 1).to(device)

def main():
    FINAL_DIR.mkdir(exist_ok=True, parents=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    lp = lpips.LPIPS(net='alex').to(device).eval()

    refs = sorted([p for p in REF_DIR.glob("*.jpg")] + [p for p in REF_DIR.glob("*.jpeg")])

    if not refs:
        print("No JPG/JPEG files in", REF_DIR)
        return

    for i, ref_path in enumerate(refs, 1):
        stem = ref_path.stem
        # Match either .jpeg or .png gens depending on your pipeline
        candidates = [
            GEN_DIR / f"{stem}_gen.jpeg",
            GEN_DIR / f"{stem}_gen.jpg",
            GEN_DIR / f"{stem}_gen.png",
        ]
        gen_path = next((c for c in candidates if c.exists()), None)
        if gen_path is None:
            print(f"[MISS] No generated match for {stem}")
            continue

        # --- Load & match preprocess exactly as generator ---
        ref_pil = resize_full_like_gen(load_rgb_pil(ref_path))
        gen_pil = load_rgb_pil(gen_path).resize(ref_pil.size, Image.LANCZOS)

        ref = to_np(ref_pil)
        gen = to_np(gen_pil)

        if SAT_MATCH:
            ref, gen = match_sat(ref, gen)

        # --- SSIM ---
        ssim_val = ssim(
            cv2.cvtColor(ref, cv2.COLOR_RGB2GRAY),
            cv2.cvtColor(gen, cv2.COLOR_RGB2GRAY),
            data_range=255,
        )

        # --- LPIPS ---
        with torch.no_grad():
            lp_val = float(lp(to_lpips_tensor(ref, device), to_lpips_tensor(gen, device)).cpu().item())

        passed = (ssim_val >= THRESHOLD_SSIM) and (lp_val <= THRESHOLD_LPIPS)

        status = "PASS" if passed else "FAIL"
        print(f"[{i}] {status}: {stem}  SSIM={ssim_val:.3f}  LPIPS={lp_val:.3f}")

        if passed:
            shutil.copy2(ref_path, FINAL_DIR / ref_path.name)
            shutil.copy2(gen_path, FINAL_DIR / gen_path.name)


if __name__ == "__main__":
    main()
