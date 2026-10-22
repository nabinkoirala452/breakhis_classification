# =============================================================================
# 04 - Small Vision-Language Models (VLMs) on BreaKHis  (binary + 8-class)
# =============================================================================
# Kaggle setup:
#   * Accelerator : GPU T4 x2
#   * Internet    : ON   (models are downloaded from the Hugging Face Hub;
#                         every model below is un-gated, no HF token needed)
#   * Inputs      : the BreaKHis image dataset (same one used by notebook 01)
#                   + the notebook-01 artifact datasets (samples_all.pkl, folds_all.npy)
#                     for binary and multiclass, so the VLMs are scored on the
#                     EXACT same patient-disjoint 5 folds as ZoomMoE / baselines.
#
# Paste each "# %% [Cell N]" block into its own notebook cell, run top to bottom.
#
# Three VLM settings are evaluated (standard in the VLM-for-pathology literature):
#   A. Zero-shot contrastive VLMs   (CLIP, SigLIP, PLIP, QuiltNet, BiomedCLIP)
#      -> image/text similarity with prompt ensembles, no training at all.
#   B. Linear probe on the frozen image encoders of the same VLMs
#      -> logistic regression trained on each fold's train split
#         (C picked on the fold's val split), tested on the fold's test split.
#   C. Zero-shot generative small VLMs (Qwen2-VL-2B, Qwen2.5-VL-3B,
#      SmolVLM2-2.2B, SmolVLM-500M)
#      -> multiple-choice prompt; the class probability is the softmax over the
#         next-token logits of the option letters (no text parsing, gives a
#         proper probability so ROC-AUC is defined).
#
# Every image is scored once; a multi-scale sample (patient, image_id) gets the
# mean probability over its available magnifications, which is the same unit
# ZoomMoE predicts on. Per-magnification image-level scores are reported too.
# All intermediate outputs are cached in /kaggle/working, so a restarted
# session resumes where it stopped.
#
# Memory: biggest model is Qwen2.5-VL-3B in fp16 (~7.5 GB) -> one replica per
# T4 (16 GB). Generative models run one replica on each GPU, data split in two.
# =============================================================================


# %% [Cell 1] Install pinned libraries (run first, before any transformers import)
import subprocess
import sys

subprocess.run(
    [
        sys.executable, "-m", "pip", "install", "-q",
        "transformers==4.51.3",      # supports Qwen2-VL, Qwen2.5-VL, SmolVLM/SmolVLM2
        "accelerate>=0.34",
        "open_clip_torch==2.32.0",   # BiomedCLIP / QuiltNet
        "num2words",                 # required by the SmolVLM2 processor
    ],
    check=True,
)


# %% [Cell 2] Imports and configuration
import gc
import json
import os
import random
import re
import threading
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"   # download bars render as big blank gaps in Kaggle

SEED = 42
TASKS = ["binary", "multiclass"]

# notebook-01 outputs (same paths your notebook 02 reads)
ARTIFACT_DIRS = {
    "binary": Path("/kaggle/input/datasets/nabinkoirala452/articafts/artifacts"),
    "multiclass": Path("/kaggle/input/datasets/nabinkoirala452/articafts-notebook1-mc/artifacts"),
}
# only used as a fallback to rebuild the folds if the artifacts are missing
CODE_DIRS = {
    "binary": Path("/kaggle/input/datasets/nabinkoirala452/code-files"),
    "multiclass": Path("/kaggle/input/datasets/nabinkoirala452/code-files-mc"),
}
# optional: ZoomMoE per-fold CSVs from notebook 02, merged into the final table
ZOOMMOE_RESULTS = {
    "binary": None,  # e.g. Path("/kaggle/input/<your-nb02-output>/artifacts/zoommoe_results.csv")
    "multiclass": None,  # e.g. Path(".../artifacts/zoommoe_multiclass_results.csv")
}

RUN_CONTRASTIVE = True   # settings A + B
RUN_GENERATIVE = True    # setting C
QUICK = False            # True -> ~1/8 of the images, to check the pipeline end-to-end

# QUICK runs write to their own folder so their cache never leaks into the full run
OUT = Path("/kaggle/working/vlm_results_quick" if QUICK else "/kaggle/working/vlm_results")
CACHE = OUT / "cache"
CACHE.mkdir(parents=True, exist_ok=True)

N_GPUS = torch.cuda.device_count()
DEVICES = [torch.device(f"cuda:{i}") for i in range(N_GPUS)] or [torch.device("cpu")]
DTYPE = torch.float16    # T4 has no native bf16 -> fp16

MAGS = (40, 100, 200, 400)
MC_CLASSES = [
    "Adenosis", "Fibroadenoma", "Phyllodes Tumor", "Tubular Adenoma",
    "Ductal Carcinoma", "Lobular Carcinoma", "Mucinous Carcinoma", "Papillary Carcinoma",
]
BIN_CLASSES = ["Benign", "Malignant"]
CLASS_NAMES = {"binary": BIN_CLASSES, "multiclass": MC_CLASSES}
SUBTYPE_TO_CLASS = {"A": 0, "F": 1, "PT": 2, "TA": 3, "DC": 4, "LC": 5, "MC": 6, "PC": 7}


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


set_seed()
print("torch", torch.__version__, "| GPUs:", [torch.cuda.get_device_name(i) for i in range(N_GPUS)])
import transformers, open_clip  # noqa: E402
print("transformers", transformers.__version__, "| open_clip", open_clip.__version__)


# %% [Cell 3] Load the notebook-01 samples and patient-disjoint folds
def load_bc(task):
    """Import that task's breakhis_common.py under a unique module name."""
    import importlib.util

    path = CODE_DIRS[task] / "breakhis_common.py"
    spec = importlib.util.spec_from_file_location(f"bc_{task}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_task(task):
    art = ARTIFACT_DIRS[task]
    if (art / "samples_all.pkl").exists() and (art / "folds_all.npy").exists():
        samples = pd.read_pickle(art / "samples_all.pkl")
        folds = list(np.load(art / "folds_all.npy", allow_pickle=True))
        print(f"[{task}] loaded artifacts from {art}")
    else:
        # identical to notebook 01 (deterministic with seed 42)
        bc = load_bc(task)
        df = bc.scan_breakhis(bc.find_dataset_root())
        samples = bc.build_groups(df, pair_mode="image_id", drop_incomplete=False)
        folds = bc.make_folds(samples, n_splits=5, val_frac=0.15, seed=SEED)
        print(f"[{task}] artifacts not found -> rebuilt folds with breakhis_common")
    return samples.reset_index(drop=True), folds


def remap_paths(samples_by_task):
    """If the BreaKHis dataset is mounted at a different path, remap by filename."""
    first = next(iter(samples_by_task.values()))
    probe = next(p for lst in first["cand40"].tolist() + first["cand400"].tolist() for p in lst)
    if Path(probe).exists():
        return samples_by_task
    print("image paths in the pickles do not exist here -> remapping by filename")
    index = {p.name: str(p) for p in Path("/kaggle/input").rglob("SOB_*.png")}
    for s in samples_by_task.values():
        for m in MAGS:
            s[f"cand{m}"] = s[f"cand{m}"].apply(lambda lst: [index[Path(p).name] for p in lst])
    return samples_by_task


_loaded = {t: load_task(t) for t in TASKS}
_samples = remap_paths({t: _loaded[t][0] for t in TASKS})
TASK = {}
for t in TASKS:
    folds = _loaded[t][1]
    TASK[t] = {"samples": _samples[t], "folds": folds}
    s = TASK[t]["samples"]
    # every patient must stay in one split
    for i, f in enumerate(folds):
        sets = {k: set(s.iloc[v].patient_id) for k, v in f.items()}
        assert not (sets["train"] & sets["test"]) and not (sets["val"] & sets["test"]), f"leak in fold {i}"
    print(f"[{t}] samples={len(s)} patients={s.patient_id.nunique()} folds={len(folds)} "
          f"label counts={s.label.value_counts().sort_index().tolist()}")

# sanity: labels agree with the filename-derived subtype
_s = TASK["multiclass"]["samples"]
assert (_s.label.values == _s.tumor_type.map(SUBTYPE_TO_CLASS).values).all()
_s = TASK["binary"]["samples"]
assert (_s.label.values == _s.tumor_type.isin(["DC", "LC", "MC", "PC"]).astype(int).values).all()


# %% [Cell 4] Image table (each image scored once, shared by both tasks)
all_paths = sorted({
    p for t in TASKS for m in MAGS for lst in TASK[t]["samples"][f"cand{m}"] for p in lst
})
IMG = pd.DataFrame({"filepath": all_paths})
IMG["mag"] = IMG.filepath.map(lambda p: int(Path(p).stem.split("-")[-2]))
IMG["patient_id"] = IMG.filepath.map(lambda p: "-".join(Path(p).stem.split("-")[:3]))
IMG["tumor_type"] = IMG.filepath.map(lambda p: Path(p).stem.split("-")[0].split("_")[-1])
PATH2IDX = {p: i for i, p in enumerate(IMG.filepath)}

if QUICK:
    # keep ALL patients (so every fold keeps its train/val/test patients),
    # but only every 8th sample of each patient
    for t in TASKS:
        s = TASK[t]["samples"]
        keep = (s.groupby("patient_id").cumcount() % 8 == 0).values
        new_idx = -np.ones(len(s), dtype=int)
        new_idx[keep] = np.arange(keep.sum())
        TASK[t]["samples"] = s[keep].reset_index(drop=True)
        TASK[t]["folds"] = [{k: new_idx[v][new_idx[v] >= 0] for k, v in f.items()} for f in TASK[t]["folds"]]
    kept = {p for t in TASKS for m in MAGS for lst in TASK[t]["samples"][f"cand{m}"] for p in lst}
    IMG = IMG[IMG.filepath.isin(kept)].reset_index(drop=True)
    PATH2IDX = {p: i for i, p in enumerate(IMG.filepath)}
    print("QUICK mode:", {t: [len(f["test"]) for f in TASK[t]["folds"]] for t in TASKS}, "test samples per fold")

for t in TASKS:
    s = TASK[t]["samples"]
    # image indices belonging to each multi-scale sample
    TASK[t]["sample_imgs"] = [
        np.array([PATH2IDX[p] for m in MAGS for p in row[f"cand{m}"]], dtype=int)
        for _, row in s.iterrows()
    ]
    # image-level label (sample label propagated to its images)
    lab = np.full(len(IMG), -1)
    for si, idx in enumerate(TASK[t]["sample_imgs"]):
        lab[idx] = s.label.iloc[si]
    TASK[t]["img_label"] = lab

print(f"images to score: {len(IMG)}")
print(IMG.groupby("mag").size().to_dict())


# %% [Cell 5] Metrics (same definitions as breakhis_common.evaluate) + aggregation
def patient_recognition_rate(y_true, y_pred, patient_ids):
    d = pd.DataFrame({"y": y_true, "p": y_pred, "pid": patient_ids})
    d["c"] = (d.y == d.p).astype(float)
    return float(d.groupby("pid").c.mean().mean())


def evaluate(y_true, prob, patient_ids):
    """prob: (N, C). Binary -> uses P(malignant), threshold 0.5. Multiclass -> argmax, OVR AUC."""
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob, dtype=float)
    if prob.shape[1] == 2:
        p1 = prob[:, 1]
        y_pred = (p1 > 0.5).astype(int)
        auc = roc_auc_score(y_true, p1) if len(np.unique(y_true)) > 1 else np.nan
    else:
        y_pred = prob.argmax(1)
        present = np.unique(y_true)
        if len(present) == prob.shape[1]:
            auc = roc_auc_score(y_true, prob, multi_class="ovr", labels=np.arange(prob.shape[1]))
        else:
            aucs = [roc_auc_score((y_true == c).astype(int), prob[:, c])
                    for c in present if len(np.unique(y_true == c)) > 1]
            auc = float(np.mean(aucs)) if aucs else np.nan
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "prr": patient_recognition_rate(y_true, y_pred, patient_ids),
        "auc": float(auc),
    }


def sample_level_probs(task, img_prob, sample_idx):
    """Mean image probability over the magnifications available for each sample."""
    imgs = TASK[task]["sample_imgs"]
    return np.stack([img_prob[imgs[i]].mean(0) for i in sample_idx])


def score_all_folds(task, model_name, setting, img_prob_per_fold):
    """img_prob_per_fold: one (N_IMG, C) array (fold-independent) or a list of them."""
    s = TASK[task]["samples"]
    rows = []
    for fi, fold in enumerate(TASK[task]["folds"]):
        img_prob = img_prob_per_fold[fi] if isinstance(img_prob_per_fold, list) else img_prob_per_fold
        te = fold["test"]
        if len(te) == 0:
            continue
        # main: sample level (same unit as ZoomMoE)
        sp = sample_level_probs(task, img_prob, te)
        r = evaluate(s.label.values[te], sp, s.patient_id.values[te])
        rows.append({"task": task, "model": model_name, "setting": setting, "fold": fi,
                     "level": "sample (mag-avg)", **r})
        # per magnification, image level
        te_imgs = np.concatenate([TASK[task]["sample_imgs"][i] for i in te])
        for m in MAGS:
            ii = te_imgs[IMG.mag.values[te_imgs] == m]
            if len(ii) == 0:
                continue
            r = evaluate(TASK[task]["img_label"][ii], img_prob[ii], IMG.patient_id.values[ii])
            rows.append({"task": task, "model": model_name, "setting": setting, "fold": fi,
                         "level": f"image {m}x", **r})
    return rows


ALL_ROWS = []
RES_FILE = OUT / "vlm_fold_results.csv"
if RES_FILE.exists():
    ALL_ROWS = pd.read_csv(RES_FILE).to_dict("records")
    print(f"resuming: {len(ALL_ROWS)} result rows already on disk")


def add_rows(rows):
    global ALL_ROWS
    keys = {(r["task"], r["model"], r["setting"]) for r in rows}
    ALL_ROWS = [r for r in ALL_ROWS if (r["task"], r["model"], r["setting"]) not in keys] + rows
    pd.DataFrame(ALL_ROWS).to_csv(RES_FILE, index=False)
    d = pd.DataFrame(rows)
    d = d[d.level == "sample (mag-avg)"]
    print(f"  -> {d.model.iloc[0]} | {d.setting.iloc[0]} | {d.task.iloc[0]}: "
          + "  ".join(f"{k}={d[k].mean():.4f}±{d[k].std():.4f}" for k in ["acc", "prr", "f1_macro", "auc"]))


# %% [Cell 6] Prompts for the contrastive (CLIP-style) models
TEMPLATES = [
    "a histopathology image of {}.",
    "an H&E stained microscopy image of {}.",
    "a hematoxylin and eosin stained breast biopsy showing {}.",
    "a microscopic image of breast tissue with {}.",
    "a pathology slide of {}.",
]
CLASS_TEXT = {
    "binary": [
        ["benign breast tissue", "a benign breast tumor", "a non-cancerous breast lesion"],
        ["malignant breast tissue", "breast cancer", "invasive breast carcinoma"],
    ],
    "multiclass": [
        ["breast adenosis", "adenosis, a benign breast lesion"],
        ["breast fibroadenoma", "fibroadenoma, a benign breast tumor"],
        ["phyllodes tumor of the breast", "a phyllodes tumor"],
        ["tubular adenoma of the breast", "a tubular adenoma"],
        ["ductal carcinoma of the breast", "invasive ductal carcinoma"],
        ["lobular carcinoma of the breast", "invasive lobular carcinoma"],
        ["mucinous carcinoma of the breast", "mucinous (colloid) carcinoma"],
        ["papillary carcinoma of the breast", "a papillary carcinoma"],
    ],
}


def class_prompts(task):
    """list (per class) of all template x synonym prompts"""
    return [[tpl.format(s) for s in syns for tpl in TEMPLATES] for syns in CLASS_TEXT[task]]


# %% [Cell 7] Contrastive VLM wrappers (HF transformers + open_clip)
CONTRASTIVE_MODELS = {
    # name                : (library, hub id)
    "CLIP ViT-B/16":        ("hf", "openai/clip-vit-base-patch16"),
    "SigLIP B/16":          ("hf", "google/siglip-base-patch16-224"),
    "PLIP":                 ("hf", "vinid/plip"),
    "QuiltNet B/32":        ("open_clip", "hf-hub:wisdomik/QuiltNet-B-32"),
    "BiomedCLIP":           ("open_clip", "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"),
}


def _feat(out):
    """transformers versions differ in what get_*_features returns."""
    if torch.is_tensor(out):
        return out
    for k in ("image_embeds", "text_embeds", "pooler_output"):
        if getattr(out, k, None) is not None:
            return getattr(out, k)
    return out[0]


class ContrastiveVLM:
    def __init__(self, lib, hub_id, device):
        self.lib, self.device = lib, device
        self.is_siglip = "siglip" in hub_id
        if lib == "open_clip":
            self.model, self.preprocess = open_clip.create_model_from_pretrained(hub_id)
            self.tokenizer = open_clip.get_tokenizer(hub_id)
        else:
            from transformers import AutoModel, AutoProcessor

            self.model = AutoModel.from_pretrained(hub_id, torch_dtype=torch.float32)
            self.processor = AutoProcessor.from_pretrained(hub_id)
            self.preprocess = lambda im: self.processor(images=im, return_tensors="pt")["pixel_values"][0]
        self.model = self.model.to(device).eval()
        self.logit_scale = float(self.model.logit_scale.exp().item())
        lb = getattr(self.model, "logit_bias", None)
        self.logit_bias = float(lb.item()) if lb is not None else 0.0
        self.n_params = sum(p.numel() for p in self.model.parameters())

    @torch.inference_mode()
    def encode_images(self, pixel_values):
        x = pixel_values.to(self.device)
        with torch.autocast("cuda", dtype=DTYPE, enabled=self.device.type == "cuda"):
            if self.lib == "open_clip":
                f = self.model.encode_image(x)
            else:
                f = _feat(self.model.get_image_features(pixel_values=x))
        return F.normalize(f.float(), dim=-1)

    @torch.inference_mode()
    def encode_texts(self, texts):
        if self.lib == "open_clip":
            f = self.model.encode_text(self.tokenizer(texts).to(self.device))
        else:
            tok = self.processor.tokenizer(
                texts, return_tensors="pt", truncation=True,
                padding="max_length" if self.is_siglip else True,
                max_length=64 if self.is_siglip else 77,
            ).to(self.device)
            f = _feat(self.model.get_text_features(**tok))
        return F.normalize(f.float(), dim=-1)


class ImageDS(torch.utils.data.Dataset):
    def __init__(self, paths, preprocess):
        self.paths, self.preprocess = paths, preprocess

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.preprocess(Image.open(self.paths[i]).convert("RGB"))


def safe_name(name):
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")


def cached(path):
    """True only if the cache file exists AND has one row per image of this run."""
    if not path.exists():
        return False
    if np.load(path, mmap_mode="r").shape[0] != len(IMG):
        print(f"  stale cache {path.name} (wrong number of images) -> recomputing")
        path.unlink()
        return False
    return True


# %% [Cell 8] Settings A + B: zero-shot and linear probe for contrastive VLMs
def linear_probe(task, emb):
    """Per fold: LR on train images, C chosen on val samples (macro-F1), scored on test."""
    s = TASK[task]["samples"]
    n_cls = len(CLASS_NAMES[task])
    lab = TASK[task]["img_label"]
    probs_per_fold, chosen = [], []
    for fi, fold in enumerate(TASK[task]["folds"]):
        tr_imgs = np.concatenate([TASK[task]["sample_imgs"][i] for i in fold["train"]])
        scaler = StandardScaler().fit(emb[tr_imgs])
        Xtr, Xall = scaler.transform(emb[tr_imgs]), scaler.transform(emb)
        best = (-1, None, None)
        # no val samples (can only happen on tiny subsets) -> fixed C
        grid = (0.001, 0.01, 0.1, 1.0) if len(fold["val"]) else (0.01,)
        for C in grid:
            clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced")
            clf.fit(Xtr, lab[tr_imgs])
            p = np.zeros((len(IMG), n_cls))
            p[:, clf.classes_] = clf.predict_proba(Xall)
            if len(grid) == 1:
                best = (0, C, p)
                break
            sp = sample_level_probs(task, p, fold["val"])
            f1 = f1_score(s.label.values[fold["val"]], sp.argmax(1), average="macro")
            if f1 > best[0]:
                best = (f1, C, p)
        chosen.append(best[1])
        probs_per_fold.append(best[2])
    print(f"    linear-probe C per fold: {chosen}")
    return probs_per_fold


if RUN_CONTRASTIVE:
    for name, (lib, hub_id) in CONTRASTIVE_MODELS.items():
        emb_file = CACHE / f"emb_{safe_name(name)}.npy"
        t0 = time.time()
        try:
            vlm = ContrastiveVLM(lib, hub_id, DEVICES[0])
        except Exception as e:
            print(f"!! could not load {name}: {e}")
            continue
        print(f"\n=== {name}  ({vlm.n_params / 1e6:.0f}M params) ===")

        if cached(emb_file):
            emb = np.load(emb_file)
        else:
            dl = torch.utils.data.DataLoader(
                ImageDS(IMG.filepath.tolist(), vlm.preprocess),
                batch_size=128, num_workers=4, pin_memory=True,
            )
            emb = np.concatenate([vlm.encode_images(x).cpu().numpy() for x in dl])
            np.save(emb_file, emb)
        print(f"  embeddings {emb.shape}  ({time.time() - t0:.0f}s)")

        for task in TASKS:
            # A. zero-shot: prompt-ensembled class embeddings
            txt = torch.stack([vlm.encode_texts(p).mean(0) for p in class_prompts(task)])
            txt = F.normalize(txt, dim=-1).cpu().numpy()
            logits = vlm.logit_scale * emb @ txt.T + vlm.logit_bias
            zs_prob = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
            np.save(CACHE / f"prob_zs_{safe_name(name)}_{task}.npy", zs_prob)
            add_rows(score_all_folds(task, name, "zero-shot", zs_prob))

            # B. linear probe on frozen image embeddings
            lp_probs = linear_probe(task, emb)
            add_rows(score_all_folds(task, name, "linear-probe", lp_probs))

        del vlm
        free_gpu()


# %% [Cell 9] Generative small VLMs: loading + multiple-choice letter scoring
from transformers import AutoModelForImageTextToText, AutoProcessor  # noqa: E402

GENERATIVE_MODELS = {
    # name               : (hub id, batch size per GPU)
    "Qwen2-VL-2B":         ("Qwen/Qwen2-VL-2B-Instruct", 8),
    "Qwen2.5-VL-3B":       ("Qwen/Qwen2.5-VL-3B-Instruct", 4),
    "SmolVLM2-2.2B":       ("HuggingFaceTB/SmolVLM2-2.2B-Instruct", 8),
    "SmolVLM-500M":        ("HuggingFaceTB/SmolVLM-500M-Instruct", 16),
}
LETTERS = "ABCDEFGH"
BENIGN_SET = {0, 1, 2, 3}


def mcq_prompt(task, mag):
    head = (f"This is an H&E-stained microscopy image of a breast tumor biopsy "
            f"at {mag}X magnification.\n")
    if task == "binary":
        opts = "A. Benign\nB. Malignant"
        q = "Is this breast tumor benign or malignant?"
    else:
        opts = "\n".join(
            f"{LETTERS[i]}. {c} ({'benign' if i in BENIGN_SET else 'malignant'})"
            for i, c in enumerate(MC_CLASSES)
        )
        q = "Which histological subtype of breast tumor is shown?"
    return f"{head}{q}\nOptions:\n{opts}\nAnswer with the option letter only."


class GenerativeVLM:
    def __init__(self, hub_id, device, dtype=DTYPE, shard=False):
        self.device = device
        kw = {}
        if "Qwen" in hub_id:
            # 448x448 max -> ~256 visual tokens; BreaKHis images are 700x460
            kw = dict(min_pixels=224 * 224, max_pixels=448 * 448)
        self.processor = AutoProcessor.from_pretrained(hub_id, **kw)
        self.processor.tokenizer.padding_side = "left"
        ip = getattr(self.processor, "image_processor", None)
        if ip is not None and hasattr(ip, "do_image_splitting"):
            ip.do_image_splitting = False   # SmolVLM: one 384/512 tile instead of up to 17
        if shard:
            # one copy split over both T4s (used only for the fp32 fallback of >3B models)
            self.model = AutoModelForImageTextToText.from_pretrained(
                hub_id, torch_dtype=dtype, attn_implementation="sdpa", device_map="auto",
            ).eval()
        else:
            self.model = AutoModelForImageTextToText.from_pretrained(
                hub_id, torch_dtype=dtype, attn_implementation="sdpa",
            ).to(device).eval()
        self.n_params = sum(p.numel() for p in self.model.parameters())
        tok = self.processor.tokenizer
        # token ids of "A", " A", ... (both spellings, whichever is a single token)
        self.letter_ids = []
        for L in LETTERS:
            ids = set()
            for s in (L, " " + L):
                t = tok.encode(s, add_special_tokens=False)
                if len(t) == 1:
                    ids.add(t[0])
            self.letter_ids.append(sorted(ids))

    @torch.inference_mode()
    def score(self, images, prompts, n_opts):
        msgs = [[{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": p}]}]
                for p in prompts]
        texts = [self.processor.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
                 for m in msgs]
        inputs = self.processor(text=texts, images=[[im] for im in images],
                                return_tensors="pt", padding=True).to(self.device)
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(self.model.dtype)
        logits = self.model(**inputs).logits[:, -1, :].float()
        opt = torch.stack([torch.logsumexp(logits[:, self.letter_ids[k]], dim=1)
                           for k in range(n_opts)], dim=1)
        return torch.softmax(opt, dim=1).cpu().numpy()


def _finite(m):
    """fp16 overflow check on a couple of real images"""
    im = [Image.open(p).convert("RGB") for p in IMG.filepath.iloc[:2]]
    return np.isfinite(m.score(im, [mcq_prompt("binary", 40)] * 2, 2)).all()


def load_replicas(hub_id):
    reps = []
    for dev in DEVICES:
        m = GenerativeVLM(hub_id, dev, DTYPE)
        if _finite(m):
            reps.append(m)
            continue
        fp32_gb = m.n_params * 4 / 1e9
        del m, reps
        free_gpu()
        if fp32_gb < 12:
            print("  fp16 produced NaN -> fp32, one replica per GPU")
            return [GenerativeVLM(hub_id, d, torch.float32) for d in DEVICES]
        if len(DEVICES) > 1:
            print("  fp16 produced NaN -> fp32, one copy sharded over both GPUs")
            return [GenerativeVLM(hub_id, DEVICES[0], torch.float32, shard=True)]
        raise RuntimeError("fp16 gives NaN and fp32 does not fit")
    return reps


def run_generative(reps, task, batch_size):
    n_opts = len(CLASS_NAMES[task])
    probs = np.zeros((len(IMG), n_opts), dtype=np.float32)
    order = np.argsort(IMG.mag.values, kind="stable")       # same mag -> same prompt length
    shards = [order[i::len(reps)] for i in range(len(reps))]
    errors = []

    def worker(rep, idx, wid):
        try:
            for b in range(0, len(idx), batch_size):
                ii = idx[b:b + batch_size]
                ims = [Image.open(IMG.filepath.iloc[j]).convert("RGB") for j in ii]
                prs = [mcq_prompt(task, IMG.mag.iloc[j]) for j in ii]
                probs[ii] = rep.score(ims, prs, n_opts)
                if wid == 0 and (b // batch_size) % 50 == 0:
                    print(f"    [{task}] gpu0 {b}/{len(idx)}", flush=True)
        except Exception as e:  # surface errors from the thread
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(r, sh, k)) for k, (r, sh) in enumerate(zip(reps, shards))]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    if errors:
        raise errors[0]
    return probs


def content_free_probs(reps, task):
    """
    Contextual calibration (Zhao et al., ICML 2021): the model's answer distribution
    for content-free images (white / grey / black) with the same prompt measures its
    option bias. One row per magnification prompt. Uses no labels.
    """
    n_opts = len(CLASS_NAMES[task])
    blanks = [Image.new("RGB", (700, 460), c) for c in ((255, 255, 255), (128, 128, 128), (0, 0, 0))]
    return np.stack([reps[0].score(blanks, [mcq_prompt(task, m)] * len(blanks), n_opts).mean(0)
                     for m in MAGS])


def calibrate(prob, cf):
    w = 1.0 / np.clip(cf[[MAGS.index(m) for m in IMG.mag.values]], 1e-6, None)
    p = prob * w
    return p / p.sum(1, keepdims=True)


# %% [Cell 10] Setting C: run the generative VLMs (zero-shot, both tasks)
TIMING = {}
if RUN_GENERATIVE:
    for name, (hub_id, bs) in GENERATIVE_MODELS.items():
        todo = [t for t in TASKS
                if not cached(CACHE / f"prob_gen_{safe_name(name)}_{t}.npy")
                or not (CACHE / f"cf_gen_{safe_name(name)}_{t}.npy").exists()]
        reps = None
        if todo:
            try:
                reps = load_replicas(hub_id)
            except Exception as e:
                print(f"!! skipping {name}: {e}")
                free_gpu()
                continue
            print(f"\n=== {name}  ({reps[0].n_params / 1e9:.2f}B params, "
                  f"{torch.cuda.memory_allocated(DEVICES[0]) / 1e9:.1f} GB on gpu0) ===")
        for task in TASKS:
            f = CACHE / f"prob_gen_{safe_name(name)}_{task}.npy"
            if cached(f):
                prob = np.load(f)
            else:
                t0 = time.time()
                prob = run_generative(reps, task, bs)
                dt = time.time() - t0
                TIMING[(name, task)] = dt / len(IMG) * len(reps)  # GPU-seconds per image
                np.save(f, prob)
                print(f"  {task}: {dt / 60:.1f} min")
            cf_file = CACHE / f"cf_gen_{safe_name(name)}_{task}.npy"
            if cf_file.exists():
                cf = np.load(cf_file)
            else:
                cf = content_free_probs(reps, task)
                np.save(cf_file, cf)
            prob_cal = calibrate(prob, cf)
            np.save(CACHE / f"prob_gencal_{safe_name(name)}_{task}.npy", prob_cal)

            for tag, p in (("raw", prob), ("calibrated", prob_cal)):
                pred_hist = np.bincount(p.argmax(1), minlength=p.shape[1])
                print(f"  {task} {tag} prediction histogram: {pred_hist.tolist()}  (collapse check)")
            add_rows(score_all_folds(task, name, "zero-shot (generative)", prob))
            add_rows(score_all_folds(task, name, "zero-shot (generative, calibrated)", prob_cal))
        if reps is not None:
            del reps
            free_gpu()


# %% [Cell 11] Summary tables (mean ± std over the 5 patient-disjoint folds)
res = pd.DataFrame(ALL_ROWS)

# optional: put ZoomMoE in the same table
for task, path in ZOOMMOE_RESULTS.items():
    if path is not None and Path(path).exists():
        z = pd.read_csv(path)
        z = z[[c for c in ["acc", "prr", "f1_macro", "auc"] if c in z.columns]].copy()
        z["task"], z["model"], z["setting"], z["level"] = task, "ZoomMoE (ours)", "trained", "sample (mag-avg)"
        z["fold"] = range(len(z))
        res = pd.concat([res, z], ignore_index=True)


def fmt(g):
    return pd.Series({k: f"{g[k].mean():.4f} ± {g[k].std():.4f}" for k in ["acc", "prr", "f1_macro", "auc"]})


for task in TASKS:
    main = res[(res.task == task) & (res.level == "sample (mag-avg)")]
    tab = main.groupby(["setting", "model"], sort=False).apply(fmt)
    tab.columns = ["Acc", "Patient RR", "Macro-F1", "ROC-AUC"]
    print(f"\n######## {task.upper()} — sample level, patient-disjoint 5-fold ########")
    display(tab)
    tab.to_csv(OUT / f"vlm_summary_{task}.csv")

    # per-magnification accuracy (image level), handy for the paper's magnification table
    mag = res[(res.task == task) & (res.level != "sample (mag-avg)")]
    mag_tab = mag.groupby(["setting", "model", "level"], sort=False).acc.mean().unstack("level").round(4)
    print(f"{task}: image-level accuracy per magnification")
    display(mag_tab)
    mag_tab.to_csv(OUT / f"vlm_per_mag_acc_{task}.csv")


# %% [Cell 12] Confusion matrices (multiclass, pooled over the 5 test folds) + model card info
import matplotlib.pyplot as plt  # noqa: E402


def pooled_test_preds(task, img_prob):
    s = TASK[task]["samples"]
    ys, ps = [], []
    for fi, fold in enumerate(TASK[task]["folds"]):
        p = img_prob[fi] if isinstance(img_prob, list) else img_prob
        ps.append(sample_level_probs(task, p, fold["test"]).argmax(1))
        ys.append(s.label.values[fold["test"]])
    return np.concatenate(ys), np.concatenate(ps)


cm_sources = {}
for name in GENERATIVE_MODELS:
    f = CACHE / f"prob_gen_{safe_name(name)}_multiclass.npy"
    if f.exists():
        cm_sources[f"{name} (zero-shot)"] = np.load(f)
    f = CACHE / f"prob_gencal_{safe_name(name)}_multiclass.npy"
    if f.exists():
        cm_sources[f"{name} (calibrated)"] = np.load(f)
for name in CONTRASTIVE_MODELS:
    f = CACHE / f"prob_zs_{safe_name(name)}_multiclass.npy"
    if f.exists():
        cm_sources[f"{name} (zero-shot)"] = np.load(f)

if cm_sources:
    n = len(cm_sources)
    cols = 3
    fig, axes = plt.subplots(int(np.ceil(n / cols)), cols, figsize=(5 * cols, 4.5 * np.ceil(n / cols)))
    axes = np.atleast_1d(axes).ravel()
    short = ["A", "F", "PT", "TA", "DC", "LC", "MC", "PC"]
    for ax, (name, prob) in zip(axes, cm_sources.items()):
        y, p = pooled_test_preds("multiclass", prob)
        cm = confusion_matrix(y, p, labels=range(8), normalize="true")
        ax.imshow(cm, vmin=0, vmax=1, cmap="Blues")
        ax.set_xticks(range(8), short)
        ax.set_yticks(range(8), short)
        ax.set_title(name, fontsize=9)
        for i in range(8):
            for j in range(8):
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=6)
    for ax in axes[n:]:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(OUT / "vlm_confusion_multiclass.png", dpi=200)
    plt.show()

with open(OUT / "vlm_timing.json", "w") as fh:
    json.dump({f"{k[0]}|{k[1]}": v for k, v in TIMING.items()}, fh, indent=2)
print("all outputs in", OUT)
print(sorted(p.name for p in OUT.iterdir()))
