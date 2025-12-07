# Independent-Study-Fake-Image-Detection

This repository contains the code used in my independent study project on detecting AI-generated images using pairwise training. The pipeline has two main parts:

1. **Data generation & filtering** (Stable Diffusion XL, QA filtering)
2. **Teacher–student detector** (handcrafted pairwise features + ResNet-18 student)

The code is organized as:

- `pipe2_generator.py` – generate SDXL copies of real images
- `pipe2_QA.py` – optional quality filter to keep only good real/fake pairs
- `teacherv1.py` – build pairwise handcrafted features and Δ-PCA “teacher” space
- `studentTrainv1.py` – train a ResNet-18 “student” to imitate the teacher and predict real/fake
- `studentEval2.py` – load the trained student model and evaluate it on new folders
- `artifacts_ts1/` – saved teacher and student artifacts (features, PCA, scaler, weights, logs)

---

## 1. Environment / Dependencies

The scripts assume a Python environment with:

- `torch`, `torchvision`
- `numpy`
- `opencv-python` (`cv2`)
- `scikit-learn`
- `joblib`
- For generation: `diffusers`, `transformers`, `safetensors`
- For QA: `lpips`, `scikit-image`, `Pillow`

---

## 2. Generation & QA (SDXL)

### 2.1 `pipe2_generator.py`: SDXL image-to-image copies

**Goal:**  
Given a folder of real photos, generate SDXL “copies” that preserve composition but regenerate final details via diffusion. Each real file gets a matching `_gen.jpeg` fake.

**Key configuration (top of the file):**

- `REF_DIR` – folder of real images
- `OUT_DIR` – folder where generated images are saved
- `PROMPT` / `NEG` – optional positive / negative prompts
- `STEPS`, `STRENGTH`, `SPLIT`, `CFG_BASE`, `CFG_REF` – diffusion parameters
- `MAX_WORK_SIDE`, `CENTER_CROP_SQ` – control working resolution
- `JPEG_QUALITY`, `JPEG_SUBSAMPLING`, `JPEG_PROGRESSIVE` – output JPEG settings
- `BASE_MODEL`, `REFINER` – SDXL base and refiner model names

**Rough flow:**

1. **Load real image** from `REF_DIR` (expects `.jpg` / `.jpeg`).
2. Convert to sRGB, fix EXIF rotation, resize down so the longest side ≤ `MAX_WORK_SIDE`, and ensure dimensions are multiples of 8.
3. Run SDXL **base** img2img to partially denoise (up to `SPLIT`).
4. Run SDXL **refiner** to complete the denoising.
5. Save the generated SDXL image as  
   `OUT_DIR / f"{original_stem}_gen.jpeg"`  
   with standardized JPEG settings.

**Inputs / Outputs:**

- **Reads:** all `.jpg` / `.jpeg` in `REF_DIR`
- **Writes:** `_gen.jpeg` images in `OUT_DIR`, one per real image

---

### 2.2 `pipe2_QA.py`: SSIM + LPIPS filtering

**Goal:**  
Filter out obviously bad SDXL generations and keep only high-quality real/fake pairs.

**Key configuration:**

- `REF_DIR` – original real images
- `GEN_DIR` – generated `_gen` images from `pipe2_generator.py`
- `FINAL_DIR` – output folder for “good” pairs
- `THRESHOLD_SSIM` – minimum structural similarity (higher is better)
- `THRESHOLD_LPIPS` – maximum perceptual distance (lower is better)
- `MAX_WORK_SIDE`, `ROUND_TO_8`, `SAT_MATCH` – basic resizing and optional saturation matching

**Rough flow:**

1. Loop over all `.jpg`/`.jpeg` in `REF_DIR`.
2. For each real `stem`, look for a matching generated file:
   - `GEN_DIR / f"{stem}_gen.jpeg"`
   - or `_gen.jpg` / `_gen.png`
3. Resize ref and gen to the same size, optionally adjust saturation.
4. Compute:
   - **SSIM** between grayscale versions
   - **LPIPS** perceptual distance on RGB
5. If `SSIM ≥ THRESHOLD_SSIM` **and** `LPIPS ≤ THRESHOLD_LPIPS`, mark as **PASS**:
   - Copy both the real and generated image into `FINAL_DIR`.

**Inputs / Outputs:**

- **Reads:** real images from `REF_DIR`, generated images from `GEN_DIR`
- **Writes (PASS only):**
  - `FINAL_DIR / real_name.jpg`
  - `FINAL_DIR / real_name_gen.jpeg`

These “final” folders (`*_final`) can be used as input to the teacher stage. 
(However in my final experiment I did not do this for storage and time sake)

---

## 3. Teacher–Student Model

### 3.1 `teacherv1.py`: teacher features and Δ-PCA space

**Goal:**  
From folders of paired (real, gen) images, build:

- Handcrafted feature vectors per image
- A difference vector Δ = (fake − real) per pair
- A standardized Δ-PCA space that captures pairwise fake–real differences

**Key configuration:**

- `DATA_DIRS` – list of folders containing *paired* images
  Each folder must have real images named e.g. `name.jpg` and generated counterparts named `name_gen.jpg` / `.jpeg` / `.png`.
- `GEN_SUFFIX` – `_gen`, used to identify fakes from filenames
- `TARGET_WIDTH`, `TARGET_HEIGHT` – strict center crop size (1280×840)
- `JPEG_QUALITY`, `BLUR_SIGMA` – optional JPEG-normalization + micro-blur
- Feature flags:
  - `USE_RADIAL_FFT`, `USE_LAPLACIAN_MS`, `USE_PATCH_CONTRAST`
- `PCA_COMPONENTS` – dimensionality of the teacher latent space
- `ARTIFACTS_DIR` – where outputs are saved 

**Rough flow:**

1. **Discover pairs** in each `DATA_DIR`:
   - Real: `name.jpg`
   - Fake: `name_gen.jpg` / `.jpeg` / `.png`
2. For each pair:
   - Apply strict preprocessing: RGB load, require ≥1280×840, center-crop to exactly 1280×840, optional JPEG/blur.
   - Extract handcrafted features for real and fake.
3. Stack real features into `fr_arr`, fake features into `fg_arr`.
4. Compute Δ = `fg_arr - fr_arr`.
5. Fit:
   - `StandardScaler` on Δ
   - `PCA` on the scaled Δ with `PCA_COMPONENTS` components.
6. Save artifacts to `ARTIFACTS_DIR`.

**Outputs in `artifacts_ts1/`:**

- `teacher_features.npz`:
  - `fr_arr` – features for real images
  - `fg_arr` – features for fake images
  - `pair_ids` – string IDs per pair (for grouping)
  - `real_paths` – full paths to real images
  - `gen_paths` – full paths to fake images
- `teacher_scaler_delta.pkl` – `StandardScaler` fitted on Δ
- `teacher_pca.pkl` – `PCA` model fitted on Δ

These are consumed by `studentTrainv1.py`.

---

### 3.2 `studentTrainv1.py`: train student ResNet-18

**Goal:**  
Train a ResNet-18 “student” model that:

- Takes a **single** image (real or fake),
- Predicts:
  - A teacher-space vector `z_hat` (same dim as PCA components),
  - A scalar logit representing **probability of fake**.

The loss is a combination of:

- MSE between `z_hat` and a **teacher target z-teacher**, and
- Binary cross-entropy real/fake classification loss.

**Key configuration:**

- `ARTIFACTS_DIR` – must match the teacher script (`artifacts_ts1`)
- Preprocessing constants – must match teacher:
  - `TARGET_WIDTH`, `TARGET_HEIGHT`
  - `JPEG_QUALITY`, `BLUR_SIGMA`
- `PCA_COMPONENTS` – dimension of teacher z (must match teacher)
- Training hyperparameters:
  - `BATCH_SIZE`
  - `NUM_EPOCHS`
  - `LEARNING_RATE`
  - `BETA_REPR` – weight between representation loss and classification loss
  - `TEST_FRACTION` – fraction of images held out for test
  - `RANDOM_SEED` – reproducibility

**Rough flow:**

1. **Load teacher artifacts** from `artifacts_ts1`:
   - `teacher_features.npz` → `fr_arr`, `fg_arr`, `pair_ids`, `real_paths`, `gen_paths`
   - `teacher_scaler_delta.pkl` and `teacher_pca.pkl`
2. For each pair:
   - Compute **teacher z** separately for real and fake:
     - `z_real = PCA( scaler(fr) )`
     - `z_fake = PCA( scaler(fg) )`
   - Wrap into `StudentItem` objects:
     - Real image path, label 0, `z_real`, `pair_id`
     - Fake image path, label 1, `z_fake`, `pair_id`
3. Split into **train/test** using `GroupShuffleSplit` on `pair_id` so all images from a pair stay in the same split.
4. `StudentDataset` applies **the same** preprocessing as the teacher:
   - Read with OpenCV
   - Require ≥ 1280×840
   - Center-crop to 1280×840
   - JPEG roundtrip (if enabled)
   - Convert to tensor and normalize with ImageNet mean/std.
5. Train `StudentNet` (ResNet-18 backbone + small MLP head) for `NUM_EPOCHS`, printing per-epoch:
   - Train loss
   - Test AUC / AP / accuracy on held-out test set.
6. Save artifacts to `artifacts_ts1/`:

   - `student_resnet18_ts.pth` – student model weights (`state_dict`)
   - `train_log.json` – JSON with per-epoch losses and metrics

**Outputs in `artifacts_ts1/` (student-related):**

- `student_resnet18_ts.pth` – *needed* for evaluation
- `train_log.json` – optional, for plotting or checking training history

---

### 3.3 `studentEval2.py`: evaluate trained student

**Goal:**  
Load the trained student from `artifacts_ts1` and:

1. Evaluate on a **larger labeled SDXL set**, computing:
   - AUC
   - Average Precision
   - Accuracy
   - ROC curve (saved as PNG)
   - Per-image predictions (CSV + JSON)
2. Run on a **small folder of example images**, printing per-image fake probabilities and saving them to JSON.

**Key configuration:**

- `MODEL_CKPT` – path to `student_resnet18_ts.pth` in `artifacts_ts1`
- `BIG_EVAL_DIR` – folder with mixed real and `_gen` images (same naming pattern as training)
- `SMALL_EVAL_DIR` – folder with a handful of example images
- `GEN_SUFFIX` – `_gen` (used to assign labels: 1 = fake, 0 = real)
- Preprocessing constants – must match training/teacher:
  - `TARGET_WIDTH`, `TARGET_HEIGHT`
  - `JPEG_QUALITY`, `BLUR_SIGMA`
- Output file names:
  - `BIG_PRED_CSV` – per-image predictions for the big eval set
  - `BIG_METRICS_JSON` – AUC/AP/Acc summary
  - `BIG_ROC_PNG` – ROC curve plot
  - `SMALL_PRED_JSON` – predictions for the small example set

**Rough flow:**

1. Build a **big eval list** from `BIG_EVAL_DIR`:
   - Skip images smaller than 1280×840
   - Label fakes if filename stem ends with `_gen`, else real.
2. Build a **small eval list** from `SMALL_EVAL_DIR`:
   - Skip too-small images
   - No labels; just report probabilities.
3. `EvalDataset` uses the **same preprocessing** as training (OpenCV load, crop 1280×840, JPEG roundtrip, ImageNet normalization).
4. Load `StudentNet`, load weights from `MODEL_CKPT`.
5. For the big eval set:
   - Run model on all images and collect probabilities and labels.
   - Compute AUC, AP, Accuracy.
   - Save ROC plot and metrics JSON.
   - Save per-image predictions as CSV + JSON.
6. For the small eval set:
   - Print `prob_fake` per file to stdout.
   - Save them to `SMALL_PRED_JSON`.

**What `artifacts_ts1/` is used for here:**

- `studentEval2.py` **only needs**:
  - `student_resnet18_ts.pth` (via `MODEL_CKPT`)
- The teacher artifacts (`teacher_features.npz`, `teacher_scaler_delta.pkl`, `teacher_pca.pkl`) are not loaded in this script, but are useful if someone wants to retrain or inspect the teacher space.
- `train_log.json` is for inspecting training behavior but is not required for evaluation.
