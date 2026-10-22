"""
breakhis_common.py
==================
Shared data pipeline, patient-disjoint folds, datasets and metrics for the
ZoomMoE / BreaKHis multi-scale project.

Import this from all three notebooks so that every experiment uses identical
splits. Put it next to the notebook, or upload it as a Kaggle utility script.

Key differences from the original baseline notebook, all deliberate:

1.  patient_id is the FULL biopsy prefix ("SOB_B_TA-14-4659"), not just the
    year field. The original used parts[1] ("14"), which collapses ~82
    patients into a handful of groups and silently breaks any attempt at
    patient-disjoint splitting.
2.  Incomplete magnification groups are kept, with a boolean mask, instead of
    being dropped. The recurrent fusion model accepts variable-length zoom
    sequences.
3.  pair_mode="pool" acknowledges that BreaKHis images at different
    magnifications are NOT registered fields of view. Pairing by image_id is
    arbitrary; drawing a random image per magnification from the same slide is
    equally valid and acts as augmentation.
"""

from __future__ import annotations

import os
import random
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

MAGS = (40, 100, 200, 400)
MAG2IDX = {m: i for i, m in enumerate(MAGS)}
CLASS_NAMES = (
    "Adenosis",
    "Fibroadenoma",
    "Phyllodes Tumor",
    "Tubular Adenoma",
    "Ductal Carcinoma",
    "Lobular Carcinoma",
    "Mucinous Carcinoma",
    "Papillary Carcinoma",
)
SUBTYPE_TO_CLASS = {
    "A": 0,
    "F": 1,
    "PT": 2,
    "TA": 3,
    "DC": 4,
    "LC": 5,
    "MC": 6,
    "PC": 7,
}
CLASS_TO_SUBTYPE = {v: k for k, v in SUBTYPE_TO_CLASS.items()}
N_CLASSES = len(CLASS_NAMES)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# SOB_B_TA-14-4659-40-001  ->  biopsy / year / slide / magnification / image
FNAME_RE = re.compile(
    r"^(?P<bio>[A-Za-z]+_[BM]_[A-Za-z]+)-(?P<yr>[0-9A-Za-z]+)-(?P<slide>[0-9A-Za-z]+)"
    r"-(?P<mag>\d+)-(?P<img>\d+)$"
)


# --------------------------------------------------------------------------
# reproducibility
# --------------------------------------------------------------------------
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# --------------------------------------------------------------------------
# dataset discovery and parsing
# --------------------------------------------------------------------------
def find_dataset_root(candidates: Sequence[str] = ("/kaggle/input", "/content", ".")) -> Path:
    """Locate the BreaKHis image tree by looking for SOB_*.png files."""
    for base in candidates:
        b = Path(base)
        if not b.exists():
            continue
        for p in b.rglob("SOB_*.png"):
            for anc in p.parents:
                if anc.name.lower().startswith("breakhis"):
                    return anc
            return b
    raise FileNotFoundError(
        f"No SOB_*.png found under {list(candidates)}. "
        "Attach the BreaKHis dataset or pass an explicit root."
    )


def scan_breakhis(root: Path) -> pd.DataFrame:
    """Walk the image tree and parse every filename into a row."""
    rows, skipped = [], 0
    for f in Path(root).rglob("*.png"):
        m = FNAME_RE.match(f.stem)
        if m is None:
            skipped += 1
            continue
        bio = m.group("bio")
        rows.append(
            {
                "filepath": str(f),
                "label": SUBTYPE_TO_CLASS[bio.split("_")[-1]],
                "tumor_type": bio.split("_")[-1],
                # full biopsy identifier == one patient in BreaKHis
                "patient_id": f"{bio}-{m.group('yr')}-{m.group('slide')}",
                "mag": int(m.group("mag")),
                "image_id": int(m.group("img")),
            }
        )
    df = pd.DataFrame(rows)
    if skipped:
        print(f"[scan] skipped {skipped} files whose names did not parse")
    df = df[df["mag"].isin(MAGS)].reset_index(drop=True)
    return df


def dataset_summary(df: pd.DataFrame) -> None:
    print(f"images            : {len(df)}")
    print(f"patients (biopsies): {df.patient_id.nunique()}")
    print("patients by class  :")
    for class_id, class_name in enumerate(CLASS_NAMES):
        print(f"  {class_id}: {class_name:<20} {df[df.label == class_id].patient_id.nunique()}")
    print(f"tumour subtypes    : {sorted(df.tumor_type.unique())}")
    print("\nimages per magnification / class:")
    print(pd.crosstab(df["mag"], df["label"]))


# --------------------------------------------------------------------------
# grouping into multi-scale samples
# --------------------------------------------------------------------------
def build_groups(
    df: pd.DataFrame,
    pair_mode: str = "image_id",
    drop_incomplete: bool = False,
) -> pd.DataFrame:
    """
    Build one row per multi-scale sample.

    pair_mode="image_id"  candidates for each magnification are the single file
                          sharing that image_id (baseline-compatible).
    pair_mode="pool"      candidates are ALL images of that patient at that
                          magnification, so a random pairing can be drawn each
                          epoch. Honest about the lack of field-of-view
                          registration, and free augmentation.
    """
    assert pair_mode in {"image_id", "pool"}
    pool: Dict[tuple, List[str]] = (
        df.groupby(["patient_id", "mag"])["filepath"].apply(list).to_dict()
    )

    records = []
    for (pid, img_id), g in df.groupby(["patient_id", "image_id"]):
        rec = {
            "sample_id": f"{pid}#{img_id:03d}",
            "patient_id": pid,
            "label": int(g["label"].iloc[0]),
            "tumor_type": g["tumor_type"].iloc[0],
        }
        n = 0
        for m in MAGS:
            if pair_mode == "pool":
                cands = pool.get((pid, m), [])
            else:
                cands = g.loc[g["mag"] == m, "filepath"].tolist()
            rec[f"cand{m}"] = cands
            n += int(len(cands) > 0)
        rec["n_mags"] = n
        records.append(rec)

    samples = pd.DataFrame(records)
    if drop_incomplete:
        samples = samples[samples["n_mags"] == len(MAGS)]
    samples = samples[samples["n_mags"] > 0].reset_index(drop=True)

    codes = {p: i for i, p in enumerate(sorted(samples["patient_id"].unique()))}
    samples["patient_idx"] = samples["patient_id"].map(codes)
    return samples


# --------------------------------------------------------------------------
# patient-disjoint folds
# --------------------------------------------------------------------------
def make_folds(
    samples: pd.DataFrame,
    n_splits: int = 5,
    val_frac: float = 0.15,
    seed: int = 42,
) -> List[Dict[str, np.ndarray]]:
    """Patient-disjoint outer CV, with a patient-disjoint inner val split."""
    y = samples["label"].values
    g = samples["patient_id"].values
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    folds = []
    for tr_idx, te_idx in sgkf.split(samples, y, groups=g):
        gss = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
        inner_tr, inner_va = next(gss.split(tr_idx, y[tr_idx], groups=g[tr_idx]))
        folds.append(
            {
                "train": tr_idx[inner_tr],
                "val": tr_idx[inner_va],
                "test": te_idx,
            }
        )
    return folds


def audit_folds(samples: pd.DataFrame, folds: List[Dict[str, np.ndarray]]) -> None:
    """Fail loudly if any patient appears in more than one split."""
    for i, f in enumerate(folds):
        sets = {k: set(samples.iloc[v]["patient_id"]) for k, v in f.items()}
        for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
            overlap = sets[a] & sets[b]
            assert not overlap, f"fold {i}: patient leak between {a} and {b}: {overlap}"
        counts = {
            k: samples.iloc[v]["label"].value_counts().reindex(range(N_CLASSES), fill_value=0).tolist()
            for k, v in f.items()
        }
        print(
            f"fold {i}: n={ {k: len(v) for k, v in f.items()} }  "
            f"patients={ {k: len(s) for k, s in sets.items()} }  "
            f"class counts={counts}"
        )


def leaky_folds(samples: pd.DataFrame, n_splits: int = 5, seed: int = 42):
    """
    Deliberately WRONG splitter: random over samples, ignoring patient.
    Only used to quantify how much the original protocol inflated results.
    """
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    y = samples["label"].values
    folds = []
    for tr_idx, te_idx in skf.split(samples, y):
        cut = int(0.85 * len(tr_idx))
        folds.append({"train": tr_idx[:cut], "val": tr_idx[cut:], "test": te_idx})
    return folds


# --------------------------------------------------------------------------
# torch datasets (imported lazily so the module works without torch)
# --------------------------------------------------------------------------
def build_transforms(img_size: int = 224, train: bool = False):
    from torchvision import transforms as T

    if train:
        return T.Compose(
            [
                T.Resize((img_size, img_size)),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.RandomApply([T.RandomRotation(90)], p=0.5),
                T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.03),
                T.ToTensor(),
                T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    return T.Compose(
        [
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def make_image_dataset(
    samples: pd.DataFrame,
    train: bool = False,
    img_size: int = 224,
    scale_dropout: float = 0.0,
    seed: int = 42,
):
    """Factory so torch is only imported when actually needed."""
    import torch
    from PIL import Image
    from torch.utils.data import Dataset

    tf = build_transforms(img_size, train)

    class MultiScaleBreaKHis(Dataset):
        def __init__(self):
            self.samples = samples.reset_index(drop=True)
            self.train = train
            self.scale_dropout = scale_dropout
            self.rng = random.Random(seed)

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, i):
            row = self.samples.iloc[i]
            imgs = torch.zeros(len(MAGS), 3, img_size, img_size)
            mask = torch.zeros(len(MAGS), dtype=torch.bool)

            for j, m in enumerate(MAGS):
                cands = row[f"cand{m}"]
                if not cands:
                    continue
                # random draw while training, stable choice at eval time
                path = self.rng.choice(cands) if self.train else cands[i % len(cands)]
                imgs[j] = tf(Image.open(path).convert("RGB"))
                mask[j] = True

            if self.train and self.scale_dropout > 0:
                for j in range(len(MAGS)):
                    if mask[j] and mask.sum() > 1 and self.rng.random() < self.scale_dropout:
                        mask[j] = False

            return {
                "images": imgs,
                "mask": mask,
                "label": torch.tensor(int(row["label"]), dtype=torch.long),
                "patient_idx": torch.tensor(int(row["patient_idx"])),
                "index": torch.tensor(int(i)),
            }

    return MultiScaleBreaKHis()


def make_feature_dataset(samples: pd.DataFrame, cache: Dict[str, np.ndarray], feat_dim: int = 512):
    """Dataset over precomputed frozen-encoder features. Used for the baselines."""
    import torch
    from torch.utils.data import Dataset

    class FeatureMultiScale(Dataset):
        def __init__(self):
            self.samples = samples.reset_index(drop=True)

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, i):
            row = self.samples.iloc[i]
            feats = torch.zeros(len(MAGS), feat_dim)
            mask = torch.zeros(len(MAGS), dtype=torch.bool)
            for j, m in enumerate(MAGS):
                cands = row[f"cand{m}"]
                if not cands:
                    continue
                feats[j] = torch.from_numpy(cache[cands[i % len(cands)]])
                mask[j] = True
            return {
                "feats": feats,
                "mask": mask,
                "label": torch.tensor(int(row["label"]), dtype=torch.long),
                "patient_idx": torch.tensor(int(row["patient_idx"])),
                "index": torch.tensor(int(i)),
            }

    return FeatureMultiScale()


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def patient_recognition_rate(y_true, y_pred, patient_ids) -> float:
    """
    Standard BreaKHis patient-level score: per-patient fraction of correctly
    classified samples, averaged over patients (unweighted).
    """
    dfm = pd.DataFrame(
        {"y": np.asarray(y_true), "p": np.asarray(y_pred), "pid": np.asarray(patient_ids)}
    )
    dfm["correct"] = (dfm["y"] == dfm["p"]).astype(float)
    return float(dfm.groupby("pid")["correct"].mean().mean())


def evaluate(y_true, y_prob, patient_ids, threshold: float = 0.5) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    
    if y_prob.ndim == 1:
        # Binary classification logic
        y_pred = (y_prob > threshold).astype(int)
        auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan")
    else:
        # Multiclass classification logic
        y_pred = y_prob.argmax(axis=1)
        present = np.unique(y_true)
        
        if len(present) == y_prob.shape[1]:
            # All classes present: standard scikit-learn OVR
            auc = float(roc_auc_score(y_true, y_prob, multi_class="ovr", labels=np.arange(y_prob.shape[1])))
        elif len(present) > 1:
            # Missing classes: calculate OVR manually for present classes only
            auc_scores = []
            for c in present:
                y_bin = (y_true == c).astype(int)
                if len(np.unique(y_bin)) > 1:  # Ensure both positive and negative examples exist
                    auc_scores.append(roc_auc_score(y_bin, y_prob[:, c]))
            auc = float(np.mean(auc_scores)) if auc_scores else float("nan")
        else:
            # Only 1 class present in the entire fold (rare but possible in small datasets)
            auc = float("nan")
            
    out = {
        "acc": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "prr": patient_recognition_rate(y_true, y_pred, patient_ids),
    }
    out["auc"] = auc
    return out


def summarise_folds(rows: List[Dict[str, float]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    agg = df.agg(["mean", "std"]).T
    agg.columns = ["mean", "std"]
    return agg


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
