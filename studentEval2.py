# studentEval2.py.py
#
# Load trained student model and evaluate on:
#   1) BIG_EVAL_DIR: mixed real + *_gen fakes (compute AUC/AP/Acc + ROC curve)
#   2) SMALL_EVAL_DIR: a few examples, just print prob_fake per image.
#
# IMPORTANT: uses the SAME preprocessing as studentTrainv1.py:
#   - strict 1280x840 center-crop
#   - JPEG roundtrip
#   - ImageNet normalization

import pathlib
import json
from typing import List

import numpy as np
from PIL import Image
import cv2
import csv
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import torchvision.models as models

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    roc_curve,
)

# =========================
#          CONFIG
# =========================

# Paths
MODEL_CKPT       = pathlib.Path("artifacts_ts1/student_resnet18_ts.pth")
ARTIFACTS_DIR    = pathlib.Path("artifacts_ts1")  # just for consistency if needed

BIG_EVAL_DIR     = pathlib.Path("test_images/SDXL1")      # mixed real + *_gen
SMALL_EVAL_DIR   = pathlib.Path("test_images/Random")

GEN_SUFFIX       = "_gen"
IMG_EXTS         = {".jpg", ".jpeg", ".png"}

# Preprocessing (must match teacher/student scripts)
TARGET_WIDTH     = 1280
TARGET_HEIGHT    = 840
JPEG_QUALITY     = 92
BLUR_SIGMA       = 0.0

PCA_COMPONENTS   = 8        # teacher z-dim; student head output dim
BATCH_SIZE_EVAL  = 16
RANDOM_SEED      = 12345

BIG_PRED_CSV     = "eval_sdxl_big_predictions.csv"
BIG_METRICS_JSON = "eval_sdxl_big_metrics.json"
BIG_ROC_PNG      = "eval_sdxl_roc1.png"
SMALL_PRED_JSON  = "eval_examples_predictions.json"

# =========================
#   PREPROCESS HELPERS
# =========================

def imread_rgb_float01(path: pathlib.Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Failed to read image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return rgb

def jpeg_roundtrip(img_rgb01: np.ndarray, quality: int) -> np.ndarray:
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
    Same as in student_train_ts1.py:
      - RGB [0,1]
      - require >= TARGET_WIDTH x TARGET_HEIGHT
      - center-crop to exactly (TARGET_HEIGHT, TARGET_WIDTH, 3)
      - JPEG roundtrip (if enabled)
      - optional blur
    """
    img = imread_rgb_float01(path)
    h, w = img.shape[:2]

    if w < TARGET_WIDTH or h < TARGET_HEIGHT:
        raise RuntimeError(
            f"Image too small for {TARGET_WIDTH}x{TARGET_HEIGHT} crop: {path.name} ({w}x{h})"
        )

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

# =========================
#        DATASET
# =========================

class EvalItem:
    def __init__(self, path: pathlib.Path, label: int | None):
        self.path = path
        self.label = label

class EvalDataset(Dataset):
    def __init__(self, items: List[EvalItem]):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        img = preprocess_fixed_rgb01(it.path)  # [H,W,3] float32 [0,1]

        x = torch.from_numpy(img).permute(2, 0, 1).contiguous()   # [3,H,W]
        x = x.clamp(0.0, 1.0)

        # ImageNet normalization (same as training)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        x = (x - mean) / std

        if it.label is None:
            y = torch.tensor(-1.0, dtype=torch.float32)
        else:
            y = torch.tensor(float(it.label), dtype=torch.float32)

        return x, y, it.path.name

def eval_collate_fn(batch):
    xs, ys, names = zip(*batch)
    xs = torch.stack(xs, dim=0)
    ys = torch.stack(ys, dim=0)
    return xs, ys, list(names)

# =========================
#        MODEL
# =========================

class StudentNet(nn.Module):
    def __init__(self, k: int):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(base.children())[:-1])
        feat_dim = base.fc.in_features
        self.fc_repr = nn.Linear(feat_dim, k)
        self.fc_cls  = nn.Linear(k, 1)

    def forward(self, x):
        feat = self.backbone(x)           # [B, C, 1, 1]
        feat = feat.view(feat.size(0), -1)
        z_hat = self.fc_repr(feat)        # [B, k]
        logit = self.fc_cls(z_hat).squeeze(1)
        return z_hat, logit

# =========================
#        HELPERS
# =========================

def list_big_eval_items(root: pathlib.Path) -> list[EvalItem]:
    if not root.exists():
        print(f"[big eval] directory not found: {root}")
        return []

    files = [p for p in root.iterdir()
             if p.is_file() and p.suffix.lower() in IMG_EXTS]

    items = []
    dropped_small = 0
    for p in files:
        # Quick size check like in teacher script
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[big eval] warn: failed to read {p.name}, skipping")
            continue
        h, w = img.shape[:2]
        if w < TARGET_WIDTH or h < TARGET_HEIGHT:
            dropped_small += 1
            # optional: print first few examples
            if dropped_small <= 5:
                print(f"[big eval drop-small] {p.name} ({w}x{h})")
            continue

        stem = p.stem.lower()
        label = 1 if stem.endswith(GEN_SUFFIX) else 0
        items.append(EvalItem(p, label))

    print(f"[big eval] found {len(items)} usable images in {root} "
          f"(dropped_too_small={dropped_small})")
    return items


def list_small_eval_items(root: pathlib.Path) -> list[EvalItem]:
    if not root.exists():
        print(f"[small eval] directory not found: {root}")
        return []

    files = [p for p in root.iterdir()
             if p.is_file() and p.suffix.lower() in IMG_EXTS]

    items = []
    dropped_small = 0
    for p in files:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[small eval] warn: failed to read {p.name}, skipping")
            continue
        h, w = img.shape[:2]
        if w < TARGET_WIDTH or h < TARGET_HEIGHT:
            dropped_small += 1
            if dropped_small <= 5:
                print(f"[small eval drop-small] {p.name} ({w}x{h})")
            continue

        items.append(EvalItem(p, None))

    print(f"[small eval] found {len(items)} usable images in {root} "
          f"(dropped_too_small={dropped_small})")
    return items


# =========================
#          MAIN
# =========================

def main():
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Using device:", device)

    # Load model
    model = StudentNet(k=PCA_COMPONENTS).to(device)
    state = torch.load(MODEL_CKPT, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # --------- BIG EVAL ----------
    big_items = list_big_eval_items(BIG_EVAL_DIR)
    if big_items:
        big_ds = EvalDataset(big_items)
        big_loader = DataLoader(
            big_ds,
            batch_size=BATCH_SIZE_EVAL,
            shuffle=False,
            num_workers=0,
            pin_memory=use_cuda,
            collate_fn=eval_collate_fn,
        )

        all_probs = []
        all_labels = []
        all_names = []

        with torch.no_grad():
            for x, y, names in big_loader:
                x = x.to(device)
                y = y.to(device)
                _, logit = model(x)
                probs = torch.sigmoid(logit).cpu().numpy()
                labels = y.cpu().numpy()
                all_probs.append(probs)
                all_labels.append(labels)
                all_names.extend(names)

        probs  = np.concatenate(all_probs)
        labels = np.concatenate(all_labels)

        mask = labels >= 0
        probs  = probs[mask]
        labels = labels[mask]
        names  = [n for n, keep in zip(all_names, mask) if keep]

        try:
            auc = roc_auc_score(labels, probs)
        except ValueError:
            auc = float("nan")
        ap  = average_precision_score(labels, probs)
        acc = accuracy_score(labels, (probs >= 0.5).astype(np.float32))

        print(f"[big eval] AUC={auc:.4f}  AP={ap:.4f}  Acc={acc:.4f}")

        # ROC curve plot
        fpr, tpr, _ = roc_curve(labels, probs)
        plt.figure()
        plt.plot(fpr, tpr, label=f"ROC (AUC={auc:.3f})")
        plt.plot([0, 1], [0, 1], "k--", label="random")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve - SDXL Set")
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(BIG_ROC_PNG, dpi=200)
        plt.close()
        print(f"[big eval] saved ROC curve to {BIG_ROC_PNG}")

        # Per-image CSV
        with open(BIG_PRED_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["file", "label", "prob_fake"])
            for n, y, p in zip(names, labels, probs):
                writer.writerow([n, int(y), float(p)])
        print(f"[big eval] saved predictions to {BIG_PRED_CSV}")

        metrics = {
            "AUC": float(auc),
            "AP": float(ap),
            "Acc": float(acc),
            "N_images": int(len(labels)),
        }
        with open(BIG_METRICS_JSON, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"[big eval] saved metrics to {BIG_METRICS_JSON}")
    else:
        print("[big eval] no images, skipping")

    # --------- SMALL EVAL ----------
    small_items = list_small_eval_items(SMALL_EVAL_DIR)
    small_results = []

    if small_items:
        small_ds = EvalDataset(small_items)
        small_loader = DataLoader(
            small_ds,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=use_cuda,
            collate_fn=eval_collate_fn,
        )

        print("[small eval] predictions:")
        with torch.no_grad():
            for x, y, names in small_loader:
                x = x.to(device)
                _, logit = model(x)
                prob = torch.sigmoid(logit).item()
                name = names[0]
                print(f"  {name}\tprob_fake={prob:.4f}")
                small_results.append({"file": name, "prob_fake": float(prob)})

        with open(SMALL_PRED_JSON, "w", encoding="utf-8") as f:
            json.dump(small_results, f, indent=2)
        print(f"[small eval] saved to {SMALL_PRED_JSON}")
    else:
        print("[small eval] no images, skipping")

if __name__ == "__main__":
    main()
