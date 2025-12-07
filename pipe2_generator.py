# pipe2_generator.py
import os, gc, pathlib
from PIL import Image, ImageOps
import torch
import numpy as np
from diffusers import StableDiffusionXLImg2ImgPipeline, EulerAncestralDiscreteScheduler

# ============== Config ==============
REF_DIR   = pathlib.Path("ref_new/batch3")
OUT_DIR   = pathlib.Path("ref_new/batch3_gen")
PROMPT    = ""    # keep empty for closest copy
NEG       = "blurry, deformed, artifacts, text, watermark"

STEPS     = 40
SPLIT     = 0.86
STRENGTH  = 0.20
CFG_BASE  = 5.6
CFG_REF   = 5.0
SEED      = 12345

BASE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
REFINER    = "stabilityai/stable-diffusion-xl-refiner-1.0"

# Work-size control
MAX_WORK_SIDE   = 1280    # cap longest side; try 1280 if still slow // 1536
CENTER_CROP_SQ  = 0       # set to 1024 to force square crop (0 = disabled)

# JPEG output standardization
JPEG_QUALITY     = 92
JPEG_SUBSAMPLING = 2      # 4:2:0
JPEG_PROGRESSIVE = False
STRIP_METADATA   = True

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1,2,3")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
torch.set_float32_matmul_precision("high")
# ====================================

def round_to_8(x): return int(8 * round(float(x)/8))

def to_srgb_rgb(im: Image.Image) -> Image.Image:
    if im.mode not in ("RGB", "RGBA"):
        im = im.convert("RGB")
    if im.mode == "RGBA":
        im = Image.alpha_composite(Image.new("RGBA", im.size, (255,255,255,255)), im).convert("RGB")
    else:
        im = im.convert("RGB")
    im = ImageOps.exif_transpose(im)
    return im

def resize_for_work(im: Image.Image) -> Image.Image:
    w, h = im.size
    s = max(w, h)
    if s > MAX_WORK_SIDE:
        scale = MAX_WORK_SIDE / s
        im = im.resize((int(round(w*scale)), int(round(h*scale))), Image.LANCZOS)
    if CENTER_CROP_SQ and CENTER_CROP_SQ > 0:
        im = ImageOps.fit(im, (CENTER_CROP_SQ, CENTER_CROP_SQ), method=Image.LANCZOS, centering=(0.5,0.5))
    # SDXL likes multiples of 8
    w, h = im.size
    im = im.resize((round_to_8(w), round_to_8(h)), Image.LANCZOS)
    return im

def save_as_jpeg(image_pil, out_path,
                 quality=92, subsampling=2, progressive=False):
    # Re-wrap to drop metadata cleanly
    rgb = image_pil.convert("RGB")
    clean = Image.fromarray(np.array(rgb, copy=True))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    clean.save(
        out_path,
        format="JPEG",
        quality=quality,
        subsampling=subsampling,  # 2 = 4:2:0
        optimize=True,
        progressive=progressive,
        # DO NOT pass exif/icc_profile at all
    )

def load_pipelines(dtype, device_map, max_memory):
    base = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        BASE_MODEL, torch_dtype=dtype, use_safetensors=True,
        device_map=device_map, max_memory=max_memory
    )
    base.scheduler = EulerAncestralDiscreteScheduler.from_config(base.scheduler.config)
    base.enable_attention_slicing(); base.enable_vae_slicing(); base.enable_vae_tiling()
    try:
        base.enable_xformers_memory_efficient_attention()
    except Exception:
        pass

    refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        REFINER, torch_dtype=dtype, use_safetensors=True,
        device_map=device_map, max_memory=max_memory
    )
    refiner.scheduler = EulerAncestralDiscreteScheduler.from_config(refiner.scheduler.config)
    refiner.enable_attention_slicing(); refiner.enable_vae_slicing(); refiner.enable_vae_tiling()
    try:
        refiner.enable_xformers_memory_efficient_attention()
    except Exception:
        pass
    return base, refiner

def main():
    assert REF_DIR.exists(), f"Missing folder: {REF_DIR}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.float16 if device == "cuda" else torch.float32
    device_map = ("balanced" if device == "cuda" else None)
    max_memory = ({i: "10GiB" for i in range(torch.cuda.device_count())} if device == "cuda" else None)

    gen = torch.Generator(device)
    if SEED is not None:
        gen.manual_seed(SEED)

    base, refiner = load_pipelines(dtype, device_map, max_memory)

    srcs = sorted([p for p in REF_DIR.iterdir()
                   if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg"}])
    if not srcs:
        print("No .jpg/.jpeg files found in", REF_DIR)
        return

    total = len(srcs)
    for idx, ref_path in enumerate(srcs, 1):
        out_path = OUT_DIR / f"{ref_path.stem}_gen.jpeg"

        ref = Image.open(ref_path)
        ref = to_srgb_rgb(ref)
        ref = resize_for_work(ref)

        with torch.inference_mode():
            latents = base(
                prompt=PROMPT, negative_prompt=NEG,
                image=ref, strength=STRENGTH,
                num_inference_steps=STEPS,
                guidance_scale=CFG_BASE,
                denoising_end=SPLIT,
                output_type="latent",
                generator=gen,
            ).images[0]

            image = refiner(
                prompt=PROMPT, negative_prompt=NEG,
                image=latents, strength=1.0,
                num_inference_steps=STEPS,
                guidance_scale=CFG_REF,
                denoising_start=SPLIT,
                generator=gen,
            ).images[0]

        save_as_jpeg(image, out_path)
        print(f"[{idx}/{total}] Saved: {out_path.name}")

    base.reset_device_map(); refiner.reset_device_map()
    del base, refiner; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
