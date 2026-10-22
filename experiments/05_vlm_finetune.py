# =============================================================================
# 05 - Fine-tuning small VLM image encoders on BreaKHis  (binary + 8-class)
# =============================================================================
# Kaggle setup:
#   * Accelerator : GPU T4 x2
#   * Internet    : ON (models come from the Hugging Face Hub, all un-gated)
#   * Inputs      : BreaKHis images + the notebook-01 artifact datasets
#                   (samples_all.pkl, folds_all.npy) -> same 5 patient-disjoint
#                   folds as ZoomMoE and as the zero-shot / linear-probe run.
#                   Optional: the output of the previous VLM run
#                   (vlm_fold_results.csv) and the ZoomMoE result CSVs, so one
#                   table holds everything.
#
# Paste each "# %% [Cell N]" block into its own cell (or all in one cell).
#
# What is trained (per model, per task, per fold):
#   * the full image encoder of a contrastive VLM (CLIP, SigLIP, PLIP, QuiltNet,
#     BiomedCLIP) + a linear head
#   * the head is INITIALISED from the model's own zero-shot text embeddings
#     (prompt-ensembled class descriptions), the standard way to fine-tune a
#     CLIP-style model without throwing away its language alignment
#     (Wortsman et al., "Robust fine-tuning of zero-shot models", CVPR 2022)
#   * same augmentation and 224x224 input as ZoomMoE (breakhis_common transforms)
#   * AdamW, lr 1e-5 encoder / 1e-3 head, 1 warm-up epoch + cosine, AMP fp16,
#     class-balanced cross-entropy with label smoothing 0.1
#   * best epoch chosen on the fold's VAL split (sample-level macro-F1),
#     early stopping, then scored once on the fold's TEST split
#
# Evaluation is identical to the previous VLM script: image probabilities are
# averaged over the magnifications of each multi-scale sample (ZoomMoE's unit),
# plus image-level metrics per magnification.
#
# Both T4s are used: every (model, task, fold) is a job, one job per GPU at a
# time. Each finished job is saved, so a restarted session skips it.
# =============================================================================


# %% [Cell 1] Install pinned libraries (same versions as the zero-shot run)
import subprocess
import sys

subprocess.run(
    [
        sys.executable, "-m", "pip", "install", "-q",
        "transformers==4.51.3",
        "accelerate>=0.34",
        "open_clip_torch==2.32.0",
    ],
    check=True,
)


# %% [Cell 2] Imports and configuration
import gc
import os
import queue
import random
import re
import threading
import time
import traceback
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import torchvision.transforms.v2.functional as TF
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

SEED = 42
TASKS = ["binary", "multiclass"]

ARTIFACT_DIRS = {
    "binary": Path("/kaggle/input/datasets/nabinkoirala452/articafts/artifacts"),
    "multiclass": Path("/kaggle/input/datasets/nabinkoirala452/articafts-notebook1-mc/artifacts"),
}
ZOOMMOE_RESULTS = {
    "binary": Path("/kaggle/input/datasets/nabinkoirala452/zoomoe-results-binary-multiclass/zoommoe_results.csv"),
    "multiclass": Path("/kaggle/input/datasets/nabinkoirala452/zoomoe-results-binary-multiclass/zoommoe_multiclass_results.csv"),
}
# optional: vlm_fold_results.csv from the zero-shot / linear-probe run, uploaded
# as a Kaggle dataset -> its rows are merged into the final tables
PREV_VLM_RESULTS = None  # e.g. Path("/kaggle/input/<your-vlm-output>/vlm_fold_results.csv")

# name -> (library, hub id)   remove a line to skip that model
MODELS = {
    "QuiltNet B/32": ("open_clip", "hf-hub:wisdomik/QuiltNet-B-32"),
    "PLIP":          ("hf", "vinid/plip"),
    "BiomedCLIP":    ("open_clip", "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"),
    "CLIP ViT-B/16": ("hf", "openai/clip-vit-base-patch16"),
    "SigLIP B/16":   ("hf", "google/siglip-base-patch16-224"),
}

QUICK = False          # True -> 1/8 of the samples, 2 epochs: pipeline check (~15-20 min)

EPOCHS = 2 if QUICK else 10
PATIENCE = 3           # early stopping on val macro-F1
BATCH = 32             # per GPU; ViT-B at 224 in fp16 fits a T4 comfortably
LR_ENCODER = 1e-5
LR_HEAD = 1e-3
WEIGHT_DECAY = 0.05
LABEL_SMOOTH = 0.1
IMG_SIZE = 224
STALL_MINUTES = 30     # no log line from any job for this long -> stop waiting, build tables anyway
SAVE_MODELS = True     # best checkpoint per (model, task, fold), fp16, ~170-190 MB each

OUT = Path("/kaggle/working/vlm_ft_results_quick" if QUICK else "/kaggle/working/vlm_ft_results")
JOBS_DIR = OUT / "jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR = OUT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

N_GPUS = torch.cuda.device_count()
DEVICES = [torch.device(f"cuda:{i}") for i in range(N_GPUS)]
assert DEVICES, "no GPU found - switch the accelerator to GPU T4 x2"

MAGS = (40, 100, 200, 400)
MC_CLASSES = [
    "Adenosis", "Fibroadenoma", "Phyllodes Tumor", "Tubular Adenoma",
    "Ductal Carcinoma", "Lobular Carcinoma", "Mucinous Carcinoma", "Papillary Carcinoma",
]
CLASS_NAMES = {"binary": ["Benign", "Malignant"], "multiclass": MC_CLASSES}
SUBTYPE_TO_CLASS = {"A": 0, "F": 1, "PT": 2, "TA": 3, "DC": 4, "LC": 5, "MC": 6, "PC": 7}


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_name(name):
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")


set_seed()
import transformers, open_clip  # noqa: E402
print("torch", torch.__version__, "| transformers", transformers.__version__,
      "| open_clip", open_clip.__version__)
print("GPUs:", [torch.cuda.get_device_name(i) for i in range(N_GPUS)])


# %% [Cell 3] Samples, patient-disjoint folds, image table
def load_task(task):
    art = ARTIFACT_DIRS[task]
    samples = pd.read_pickle(art / "samples_all.pkl").reset_index(drop=True)
    folds = list(np.load(art / "folds_all.npy", allow_pickle=True))
    return samples, folds


TASK = {t: dict(zip(("samples", "folds"), load_task(t))) for t in TASKS}

# remap image paths if BreaKHis is mounted somewhere else
_probe = next(p for lst in TASK["binary"]["samples"]["cand40"] for p in lst)
if not Path(_probe).exists():
    print("remapping image paths by filename")
    _index = {p.name: str(p) for p in Path("/kaggle/input").rglob("SOB_*.png")}
    for t in TASKS:
        for m in MAGS:
            TASK[t]["samples"][f"cand{m}"] = TASK[t]["samples"][f"cand{m}"].apply(
                lambda lst: [_index[Path(p).name] for p in lst])

for t in TASKS:
    s, folds = TASK[t]["samples"], TASK[t]["folds"]
    for i, f in enumerate(folds):
        sets = {k: set(s.iloc[v].patient_id) for k, v in f.items()}
        assert not (sets["train"] & sets["test"]) and not (sets["val"] & sets["test"]) \
            and not (sets["train"] & sets["val"]), f"patient leak in fold {i}"
assert (TASK["multiclass"]["samples"].label.values
        == TASK["multiclass"]["samples"].tumor_type.map(SUBTYPE_TO_CLASS).values).all()

if QUICK:
    # all patients kept (folds intact), every 8th sample per patient
    for t in TASKS:
        s = TASK[t]["samples"]
        keep = (s.groupby("patient_id").cumcount() % 8 == 0).values
        new_idx = -np.ones(len(s), dtype=int)
        new_idx[keep] = np.arange(keep.sum())
        TASK[t]["samples"] = s[keep].reset_index(drop=True)
        TASK[t]["folds"] = [{k: new_idx[v][new_idx[v] >= 0] for k, v in f.items()}
                            for f in TASK[t]["folds"]]

all_paths = sorted({p for t in TASKS for m in MAGS
                    for lst in TASK[t]["samples"][f"cand{m}"] for p in lst})
IMG = pd.DataFrame({"filepath": all_paths})
IMG["mag"] = IMG.filepath.map(lambda p: int(Path(p).stem.split("-")[-2]))
IMG["patient_id"] = IMG.filepath.map(lambda p: "-".join(Path(p).stem.split("-")[:3]))
PATH2IDX = {p: i for i, p in enumerate(IMG.filepath)}

for t in TASKS:
    s = TASK[t]["samples"]
    TASK[t]["sample_imgs"] = [np.array([PATH2IDX[p] for m in MAGS for p in row[f"cand{m}"]], dtype=int)
                              for _, row in s.iterrows()]
    lab = np.full(len(IMG), -1)
    for si, idx in enumerate(TASK[t]["sample_imgs"]):
        lab[idx] = s.label.iloc[si]
    TASK[t]["img_label"] = lab
    print(f"[{t}] samples={len(s)} patients={s.patient_id.nunique()} "
          f"test samples per fold={[len(f['test']) for f in TASK[t]['folds']]}")
print("images:", len(IMG), IMG.groupby("mag").size().to_dict())


def imgs_of(task, sample_idx):
    return np.concatenate([TASK[task]["sample_imgs"][i] for i in sample_idx])


# %% [Cell 4] Pre-load every image once, resized to 224x224 (same resize as ZoomMoE)
def _load(p):
    return np.asarray(Image.open(p).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR))


t0 = time.time()
with ThreadPoolExecutor(8) as ex:
    IMG_ARR = np.stack(list(ex.map(_load, IMG.filepath)))
print(f"pre-loaded {IMG_ARR.shape} uint8 ({IMG_ARR.nbytes / 1e9:.2f} GB) in {time.time() - t0:.0f}s")


# Augmentation runs on the GPU, per image, with no DataLoader worker processes.
# (Forking DataLoader workers from two GPU threads can deadlock - that is what
# hung one GPU in an earlier run.) The ops and ranges are those of the
# breakhis_common train transform: RandomHorizontalFlip, RandomVerticalFlip,
# RandomApply([RandomRotation(90)], p=0.5), ColorJitter(0.15, 0.15, 0.15, 0.03).
def gpu_augment(x, rng):
    """x: B x 3 x H x W float in [0, 1] on the GPU; rng: np.random.Generator."""
    out = []
    for img in x:
        if rng.random() < 0.5:
            img = TF.horizontal_flip(img)
        if rng.random() < 0.5:
            img = TF.vertical_flip(img)
        if rng.random() < 0.5:
            img = TF.rotate(img, float(rng.uniform(-90, 90)), interpolation=InterpolationMode.NEAREST, fill=0.0)
        for op in rng.permutation(4):                      # ColorJitter applies its ops in random order
            if op == 0:
                img = TF.adjust_brightness(img, float(rng.uniform(0.85, 1.15)))
            elif op == 1:
                img = TF.adjust_contrast(img, float(rng.uniform(0.85, 1.15)))
            elif op == 2:
                img = TF.adjust_saturation(img, float(rng.uniform(0.85, 1.15)))
            else:
                img = TF.adjust_hue(img, float(rng.uniform(-0.03, 0.03)))
        out.append(img)
    return torch.stack(out)


def to_device(img_idx, dev, mean_t, std_t, rng=None):
    """uint8 images from RAM -> normalised float batch on dev (augmented if rng is given)."""
    x = torch.from_numpy(IMG_ARR[img_idx]).to(dev, non_blocking=True).permute(0, 3, 1, 2).float().div_(255.0)
    if rng is not None:
        x = gpu_augment(x, rng)
    return (x - mean_t) / std_t


def iterate(idx, batch_size, rng=None, drop_last=False):
    """index batches; shuffled when rng is given"""
    idx = np.asarray(idx)
    if rng is not None:
        idx = idx[rng.permutation(len(idx))]
    for b in range(0, len(idx), batch_size):
        ii = idx[b:b + batch_size]
        if drop_last and len(ii) < batch_size:
            break
        yield ii


# %% [Cell 5] Metrics (identical to breakhis_common.evaluate) + fold scoring
def patient_recognition_rate(y_true, y_pred, patient_ids):
    d = pd.DataFrame({"y": y_true, "p": y_pred, "pid": patient_ids})
    d["c"] = (d.y == d.p).astype(float)
    return float(d.groupby("pid").c.mean().mean())


def evaluate(y_true, prob, patient_ids):
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
    imgs = TASK[task]["sample_imgs"]
    return np.stack([img_prob[imgs[i]].mean(0) for i in sample_idx])


def score_fold(task, model_name, setting, fi, img_prob):
    s = TASK[task]["samples"]
    te = TASK[task]["folds"][fi]["test"]
    rows = []
    sp = sample_level_probs(task, img_prob, te)
    rows.append({"task": task, "model": model_name, "setting": setting, "fold": fi,
                 "level": "sample (mag-avg)",
                 **evaluate(s.label.values[te], sp, s.patient_id.values[te])})
    te_imgs = imgs_of(task, te)
    for m in MAGS:
        ii = te_imgs[IMG.mag.values[te_imgs] == m]
        if len(ii):
            rows.append({"task": task, "model": model_name, "setting": setting, "fold": fi,
                         "level": f"image {m}x",
                         **evaluate(TASK[task]["img_label"][ii], img_prob[ii], IMG.patient_id.values[ii])})
    return rows


# %% [Cell 6] Encoders + zero-shot-initialised classification head
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


class HFVision(nn.Module):
    """vision tower (+ projection) of a transformers CLIP / SigLIP model"""

    def __init__(self, vision_model, proj=None):
        super().__init__()
        self.vision_model, self.proj = vision_model, proj

    def forward(self, x):
        f = self.vision_model(pixel_values=x).pooler_output
        return self.proj(f) if self.proj is not None else f


def _normalize_from(preprocess):
    for t in preprocess.transforms:
        if isinstance(t, T.Normalize):
            return tuple(t.mean), tuple(t.std)
    return (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)


@torch.no_grad()
def load_encoder(name, task):
    """
    Returns (image encoder, class text embeddings [C, D] normalised, logit scale, mean, std).
    Text embeddings are computed on CPU, then the text tower is dropped.
    """
    lib, hub_id = MODELS[name]
    prompts = [[tpl.format(s) for s in syns for tpl in TEMPLATES] for syns in CLASS_TEXT[task]]
    if lib == "open_clip":
        model, preprocess = open_clip.create_model_from_pretrained(hub_id)
        tok = open_clip.get_tokenizer(hub_id)
        model.eval()
        txt = torch.stack([F.normalize(model.encode_text(tok(p)).float(), dim=-1).mean(0) for p in prompts])
        mean, std = _normalize_from(preprocess)
        visual = model.visual
    else:
        from transformers import AutoModel, AutoProcessor

        model = AutoModel.from_pretrained(hub_id, torch_dtype=torch.float32).eval()
        proc = AutoProcessor.from_pretrained(hub_id)
        siglip = "siglip" in hub_id
        feats = []
        for p in prompts:
            enc = proc.tokenizer(p, return_tensors="pt", truncation=True,
                                 padding="max_length" if siglip else True,
                                 max_length=64 if siglip else 77)
            f = model.get_text_features(**enc)
            f = f if torch.is_tensor(f) else f.pooler_output
            feats.append(F.normalize(f.float(), dim=-1).mean(0))
        txt = torch.stack(feats)
        mean, std = tuple(proc.image_processor.image_mean), tuple(proc.image_processor.image_std)
        visual = HFVision(model.vision_model, getattr(model, "visual_projection", None))
    scale = float(model.logit_scale.exp().item())
    return visual, F.normalize(txt, dim=-1), scale, mean, std


class FineTuneVLM(nn.Module):
    def __init__(self, visual, txt, scale):
        super().__init__()
        self.visual = visual
        self.head = nn.Linear(txt.shape[1], txt.shape[0])
        with torch.no_grad():
            self.head.weight.copy_(txt * min(scale, 100.0))   # zero-shot classifier as init
            self.head.bias.zero_()

    def forward(self, x):
        return self.head(F.normalize(self.visual(x).float(), dim=-1))


# %% [Cell 7] One training job = (model, task, fold) on one GPU
LOAD_LOCK = threading.Lock()
PRINT_LOCK = threading.Lock()
LAST_PROGRESS = [time.time()]   # updated on every log line; read by the stall watchdog


def log(*a):
    with PRINT_LOCK:
        print(*a, flush=True)
        LAST_PROGRESS[0] = time.time()


def job_file(name, task, fi):
    return JOBS_DIR / f"{safe_name(name)}__{task}__fold{fi}.npz"


@torch.no_grad()
def predict(model, img_idx, dev, n_cls, mean_t, std_t):
    model.eval()
    probs = np.zeros((len(IMG), n_cls), dtype=np.float32)
    for ii in iterate(img_idx, 128):
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(to_device(ii, dev, mean_t, std_t))
        probs[ii] = torch.softmax(logits.float(), 1).cpu().numpy()
    return probs


def train_job(name, task, fi, dev):
    torch.cuda.set_device(dev)
    tag = f"[{name} | {task} | fold {fi} | {dev}]"
    n_cls = len(CLASS_NAMES[task])
    fold = TASK[task]["folds"][fi]
    lab = TASK[task]["img_label"]
    tr, va, te = imgs_of(task, fold["train"]), imgs_of(task, fold["val"]), imgs_of(task, fold["test"])

    with LOAD_LOCK:
        visual, txt, scale, mean, std = load_encoder(name, task)
    model = FineTuneVLM(visual, txt, scale).to(dev)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    rng = np.random.default_rng(SEED + fi)               # shuffling + augmentation, per job
    mean_t = torch.tensor(mean, device=dev).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=dev).view(1, 3, 1, 1)
    lab_t = torch.from_numpy(lab).to(dev)
    steps_per_epoch = len(tr) // BATCH

    # class-balanced CE (same idea as class_weight="balanced" in the linear probe)
    counts = np.bincount(lab[tr], minlength=n_cls).astype(float)
    w = torch.tensor(counts.sum() / (n_cls * np.maximum(counts, 1)), dtype=torch.float32, device=dev)
    criterion = nn.CrossEntropyLoss(weight=w, label_smoothing=LABEL_SMOOTH)

    opt = torch.optim.AdamW(
        [{"params": model.visual.parameters(), "lr": LR_ENCODER},
         {"params": model.head.parameters(), "lr": LR_HEAD}],
        weight_decay=WEIGHT_DECAY,
    )
    total = EPOCHS * steps_per_epoch
    warm = steps_per_epoch

    def lr_lambda(step):
        if step < warm:
            return (step + 1) / warm
        return 0.5 * (1 + np.cos(np.pi * (step - warm) / max(total - warm, 1)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.amp.GradScaler("cuda")
    s = TASK[task]["samples"]

    best_f1, best_state, best_ep, bad = -1.0, None, -1, 0
    t0 = time.time()
    for ep in range(EPOCHS):
        model.train()
        losses = []
        for ii in iterate(tr, BATCH, rng, drop_last=True):
            x = to_device(ii, dev, mean_t, std_t, rng)
            y = lab_t[torch.from_numpy(ii).to(dev)]
            with torch.autocast("cuda", dtype=torch.float16):
                loss = criterion(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(loss.item())

        pv = predict(model, va, dev, n_cls, mean_t, std_t)
        sp = sample_level_probs(task, pv, fold["val"])
        f1 = f1_score(s.label.values[fold["val"]], sp.argmax(1), average="macro")
        improved = f1 > best_f1
        if improved:
            best_f1, best_ep, bad = f1, ep, 0
            best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
        else:
            bad += 1
        log(f"{tag} ep {ep + 1}/{EPOCHS} loss={np.mean(losses):.4f} val_f1={f1:.4f}"
            f"{' *' if improved else ''} ({(time.time() - t0) / 60:.1f} min)")
        if bad >= PATIENCE:
            log(f"{tag} early stop")
            break

    model.load_state_dict(best_state)
    pt = predict(model, te, dev, n_cls, mean_t, std_t)
    train_min = (time.time() - t0) / 60
    if SAVE_MODELS:
        # load with: m = FineTuneVLM(*load_encoder(name, task)[:3]); m.load_state_dict(
        #               {k: v.float() for k, v in ckpt["state_dict"].items()})
        torch.save({
            "state_dict": {k: v.half() if v.is_floating_point() else v for k, v in best_state.items()},
            "model": name, "hub_id": MODELS[name][1], "task": task, "fold": fi,
            "class_names": CLASS_NAMES[task], "mean": mean, "std": std, "img_size": IMG_SIZE,
            "best_epoch": best_ep + 1, "val_f1": best_f1,
        }, MODELS_DIR / f"{safe_name(name)}__{task}__fold{fi}.pt")
    # written last: its existence marks the job (and its checkpoint) as finished
    np.savez(job_file(name, task, fi), probs=pt, best_epoch=best_ep, val_f1=best_f1,
             train_min=train_min, n_params=n_params)
    r = score_fold(task, name, "fine-tuned", fi, pt)[0]
    log(f"{tag} DONE best_ep={best_ep + 1} val_f1={best_f1:.4f} | TEST acc={r['acc']:.4f} "
        f"f1={r['f1_macro']:.4f} prr={r['prr']:.4f} auc={r['auc']:.4f} ({train_min:.1f} min)")

    del model, best_state, opt
    gc.collect()
    torch.cuda.empty_cache()


# %% [Cell 8] Run all jobs on both GPUs (finished jobs are skipped on restart)
jobs = queue.Queue()
for name in MODELS:
    for task in TASKS:
        for fi in range(len(TASK[task]["folds"])):
            if not job_file(name, task, fi).exists():
                jobs.put((name, task, fi))
print(f"{jobs.qsize()} jobs to run on {len(DEVICES)} GPU(s)")
FAILED = []


def gpu_worker(dev):
    while True:
        try:
            name, task, fi = jobs.get_nowait()
        except queue.Empty:
            return
        try:
            train_job(name, task, fi, dev)
        except Exception as e:
            FAILED.append((name, task, fi, repr(e)))
            log(f"!! FAILED {name} | {task} | fold {fi} on {dev}: {e}\n{traceback.format_exc()}")
            gc.collect()
            torch.cuda.empty_cache()


T_START = time.time()
LAST_PROGRESS[0] = T_START
threads = [threading.Thread(target=gpu_worker, args=(d,), daemon=True) for d in DEVICES]
for th in threads:
    th.start()
# watchdog: never wait forever on a stuck GPU thread (a hang once ate a 12 h session)
while any(th.is_alive() for th in threads):
    for th in threads:
        th.join(timeout=30)
    if time.time() - LAST_PROGRESS[0] > STALL_MINUTES * 60:
        print(f"\n!! no progress for {STALL_MINUTES} min - stop waiting; unfinished jobs stay unfinished "
              f"(re-run this cell in a fresh session to do them)")
        break
print(f"\njobs finished in {(time.time() - T_START) / 3600:.2f} h; failed: {FAILED or 'none'}")


# %% [Cell 9] Score every finished model and build the summary tables
rows, info = [], []
for name in MODELS:
    for task in TASKS:
        n_f = len(TASK[task]["folds"])
        files = [job_file(name, task, fi) for fi in range(n_f)]
        if not all(f.exists() for f in files):
            print(f"incomplete: {name} | {task} ({sum(f.exists() for f in files)}/{n_f} folds)")
            continue
        for fi, f in enumerate(files):
            z = np.load(f)
            rows += score_fold(task, name, "fine-tuned", fi, z["probs"])
            info.append({"task": task, "model": name, "fold": fi, "best_epoch": int(z["best_epoch"]) + 1,
                         "val_f1": float(z["val_f1"]), "train_min": float(z["train_min"]),
                         "params_M": float(z["n_params"]) / 1e6})

res = pd.DataFrame(rows)
res.to_csv(OUT / "vlm_ft_fold_results.csv", index=False)
pd.DataFrame(info).to_csv(OUT / "vlm_ft_training_info.csv", index=False)

# merge zero-shot / linear-probe rows from the previous run and ZoomMoE, if available
full = res.copy()
if PREV_VLM_RESULTS is not None and Path(PREV_VLM_RESULTS).exists():
    full = pd.concat([pd.read_csv(PREV_VLM_RESULTS), full], ignore_index=True)
for task, path in ZOOMMOE_RESULTS.items():
    if path is not None and Path(path).exists():
        z = pd.read_csv(path)
        z = z[[c for c in ["acc", "prr", "f1_macro", "auc"] if c in z.columns]].copy()
        z["task"], z["model"], z["setting"], z["level"] = task, "ZoomMoE (ours)", "trained", "sample (mag-avg)"
        z["fold"] = range(len(z))
        full = pd.concat([full, z], ignore_index=True)
full.to_csv(OUT / "all_results_fold_level.csv", index=False)


def fmt(g):
    return pd.Series({k: f"{g[k].mean():.4f} ± {g[k].std():.4f}" for k in ["acc", "prr", "f1_macro", "auc"]})


for task in TASKS:
    main = full[(full.task == task) & (full.level == "sample (mag-avg)")]
    tab = main.groupby(["setting", "model"], sort=False).apply(fmt)
    tab.columns = ["Acc", "Patient RR", "Macro-F1", "ROC-AUC"]
    print(f"\n######## {task.upper()} — sample level, patient-disjoint 5-fold ########")
    display(tab)
    tab.to_csv(OUT / f"summary_{task}.csv")

    mag = res[(res.task == task) & (res.level != "sample (mag-avg)")]
    if len(mag):
        mag_tab = mag.groupby(["model", "level"], sort=False).acc.mean().unstack("level").round(4)
        print(f"{task}: fine-tuned, image-level accuracy per magnification")
        display(mag_tab)
        mag_tab.to_csv(OUT / f"ft_per_mag_acc_{task}.csv")

ti = pd.DataFrame(info)
if len(ti):
    print("\ntraining info (mean over folds):")
    display(ti.groupby(["task", "model"])[["best_epoch", "val_f1", "train_min", "params_M"]].mean().round(3))
print("outputs in", OUT, ":", sorted(p.name for p in OUT.iterdir()))
