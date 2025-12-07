# studentTrainv1.py
#
# Stage 2: train CNN student (ResNet-18) to imitate teacher Δ-PCA + detect fakes.
# Loads artifacts from ARTIFACTS_DIR produced by teacher_build_ts1.py.

import os
import json
import pathlib
from typing import List, Tuple

import numpy as np
import cv2

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score

import joblib
import torchvision.models as models

# ============================
#           CONFIG
# ============================

ARTIFACTS_DIR = pathlib.Path("artifacts_ts1")  # must match teacher script

# These MUST match teacherv1.py
TARGET_WIDTH  = 1280
TARGET_HEIGHT = 840
JPEG_QUALITY  = 92
BLUR_SIGMA    = 0.0

# Teacher space dimensionality (PCA_COMPONENTS)
PCA_COMPONENTS = 8

# Student training hyperparams
BATCH_SIZE  = 16
NUM_EPOCHS  = 5
LEARNING_RATE = 1e-4
BETA_REPR   = 0.5          # weight on MSE(z_hat, z_teacher) vs BCE
TEST_FRACTION = 0.2
RANDOM_SEED = 12345

# ============================
#   PREPROCESS (same crop)
# ============================

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

# ============================
#      TEACHER PROJECTION
# ============================

def teacher_z_from_features(f: np.ndarray, scaler_delta, pca) -> np.ndarray:
    """Map a base feature vector f to teacher Δ-PCA space."""
    f_scaled = scaler_delta.transform(f[None, :])
    z = pca.transform(f_scaled)[0]
    return z.astype(np.float32)

# ============================
#        STUDENT DATASET
# ============================

class StudentItem:
    def __init__(self, img_path: str, label: int, z_teacher: np.ndarray, pair_id: str):
        self.img_path = img_path
        self.label = label
        self.z_teacher = z_teacher
        self.pair_id = pair_id

class StudentDataset(Dataset):
    def __init__(self, items: List[StudentItem]):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        it = self.items[idx]
        path = pathlib.Path(it.img_path)

        img = preprocess_fixed_rgb01(path)   # [H,W,3] float32 [0,1]
        # to tensor [3,H,W]
        x = torch.from_numpy(img).permute(2, 0, 1).contiguous()  # [3, H, W]
        # optional clamp
        x = x.clamp(0.0, 1.0)
        # normalize like ImageNet
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        x = (x - mean) / std

        y = torch.tensor(float(it.label), dtype=torch.float32)
        z = torch.from_numpy(it.z_teacher).float()

        return x, y, z

def student_collate_fn(batch):
    xs, ys, zs = zip(*batch)
    xs = torch.stack(xs, dim=0)
    ys = torch.stack(ys, dim=0)
    zs = torch.stack(zs, dim=0)
    return xs, ys, zs

# ============================
#        STUDENT MODEL
# ============================

class StudentNet(nn.Module):
    def __init__(self, k: int):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(base.children())[:-1])  # up to avgpool
        feat_dim = base.fc.in_features

        self.fc_repr = nn.Linear(feat_dim, k)
        self.fc_cls  = nn.Linear(k, 1)

    def forward(self, x):
        feat = self.backbone(x)          # [B, feat_dim, 1, 1]
        feat = feat.view(feat.size(0), -1)
        z_hat = self.fc_repr(feat)       # [B, k]
        logit = self.fc_cls(z_hat).squeeze(1)  # [B]
        return z_hat, logit

# ============================
#           MAIN
# ============================

def main():
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    feats_path = ARTIFACTS_DIR / "teacher_features.npz"
    scaler_path = ARTIFACTS_DIR / "teacher_scaler_delta.pkl"
    pca_path = ARTIFACTS_DIR / "teacher_pca.pkl"

    if not feats_path.exists():
        raise SystemExit(f"Missing teacher npz: {feats_path}")
    if not scaler_path.exists() or not pca_path.exists():
        raise SystemExit("Missing scaler or PCA pickle from teacher stage.")

    data = np.load(feats_path, allow_pickle=True)
    fr_arr = data["fr_arr"]       # [N_pairs, d]
    fg_arr = data["fg_arr"]
    pair_ids = data["pair_ids"]   # [N_pairs]
    real_paths = data["real_paths"]
    gen_paths  = data["gen_paths"]

    scaler_delta = joblib.load(scaler_path)
    pca          = joblib.load(pca_path)

    n_pairs, feat_dim = fr_arr.shape
    print(f"[load] teacher features: pairs={n_pairs}, feat_dim={feat_dim}")

    # Build StudentItems (two per pair)
    items: List[StudentItem] = []
    for i in range(n_pairs):
        pid = str(pair_ids[i])
        fr = fr_arr[i]
        fg = fg_arr[i]

        z_real = teacher_z_from_features(fr, scaler_delta, pca)
        z_fake = teacher_z_from_features(fg, scaler_delta, pca)

        items.append(StudentItem(real_paths[i], 0, z_real, pid))
        items.append(StudentItem(gen_paths[i], 1, z_fake, pid))

    print(f"[data] total images for student: {len(items)} (2 per pair), teacher_dim={PCA_COMPONENTS}")

    # Group-aware train/test split by pair_id
    all_idx = np.arange(len(items))
    groups = np.array([it.pair_id for it in items])

    gss = GroupShuffleSplit(
        test_size=TEST_FRACTION, n_splits=1, random_state=RANDOM_SEED
    )
    train_idx, test_idx = next(gss.split(all_idx, groups=groups))

    train_items = [items[i] for i in train_idx]
    test_items  = [items[i] for i in test_idx]

    print(f"[split] train images: {len(train_items)}, test images: {len(test_items)}")

    train_ds = StudentDataset(train_items)
    test_ds  = StudentDataset(test_items)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Using device:", device)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # safer on Jupyter/Windows
        pin_memory=use_cuda,
        collate_fn=student_collate_fn,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=use_cuda,
        collate_fn=student_collate_fn,
    )

    model = StudentNet(k=PCA_COMPONENTS).to(device)
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_log = []

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        total_cls  = 0.0
        total_repr = 0.0
        n_batches  = 0

        for x, y, z in train_loader:
            x = x.to(device)
            y = y.to(device)
            z = z.to(device)

            optimizer.zero_grad()
            z_hat, logit = model(x)

            loss_repr = mse(z_hat, z)
            loss_cls  = bce(logit, y)
            loss = BETA_REPR * loss_repr + (1.0 - BETA_REPR) * loss_cls

            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            total_cls  += float(loss_cls.item())
            total_repr += float(loss_repr.item())
            n_batches  += 1

        avg_loss = total_loss / max(1, n_batches)
        avg_cls  = total_cls  / max(1, n_batches)
        avg_repr = total_repr / max(1, n_batches)

        print(f"[epoch {epoch}] train loss={avg_loss:.4f}  cls={avg_cls:.4f}  repr={avg_repr:.4f}")

        # Eval on held-out test set
        model.eval()
        all_probs = []
        all_labels = []
        with torch.no_grad():
            for x, y, z in test_loader:
                x = x.to(device)
                y = y.to(device)
                _, logit = model(x)
                prob = torch.sigmoid(logit)
                all_probs.append(prob.cpu().numpy())
                all_labels.append(y.cpu().numpy())

        if all_probs:
            probs  = np.concatenate(all_probs)
            labels = np.concatenate(all_labels)
            try:
                auc = roc_auc_score(labels, probs)
            except ValueError:
                auc = float("nan")
            ap  = average_precision_score(labels, probs)
            acc = accuracy_score(labels, (probs >= 0.5).astype(np.float32))
            print(f"         test AUC={auc:.4f}  AP={ap:.4f}  Acc={acc:.4f}")
        else:
            auc = ap = acc = float("nan")
            print("         test: no predictions?")

        train_log.append({
            "epoch": epoch,
            "train_loss": avg_loss,
            "train_loss_cls": avg_cls,
            "train_loss_repr": avg_repr,
            "test_auc": float(auc),
            "test_ap": float(ap),
            "test_acc": float(acc),
        })

    # Save model + log
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = ARTIFACTS_DIR / "student_resnet18_ts.pth"
    torch.save(model.state_dict(), model_path)
    print(f"[save] student weights -> {model_path}")

    log_path = ARTIFACTS_DIR / "train_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(train_log, f, indent=2)
    print(f"[save] training log -> {log_path}")

if __name__ == "__main__":
    main()
