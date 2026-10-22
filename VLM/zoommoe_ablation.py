# =============================================================================
# 07 - ZoomMoE training-time ablations, BINARY, 5 patient-disjoint folds
# =============================================================================
# Each ablation retrains ZoomMoE (ResNet-50, notebook 02) with ONE part removed,
# using notebook 02's exact recipe: zoommoe_model.py + breakhis_common.py from
# the code-files dataset, pooled pairing (samples_pool.pkl), folds_all.npy,
# stages 8/15/8 epochs, lr 3e-4/3e-4/3e-5, batch 8, scale dropout 0.15,
# ImageNet normalisation, freeze up to layer2 in stages 0-1.
#
# Variants (the overrides are the ones in notebook 02's ABLATIONS dict):
#   no_moe          moe_stages=()                         -> are the experts needed?
#   moe_mag_only    cond_state_dim=0                      -> does routing need the zoom state h_{k-1}?
#   no_contrastive  w_align=w_ordinal=w_disent=w_advmag=w_supcon=0  -> do the extra losses help?
#   full            nothing removed (same-pipeline reference + val predictions)
#   no_supcon / no_ordinal                                -> only if time is left
#
# TIME SAFETY (GPU quota is limited):
#   * jobs run in PRIORITY order: all 5 folds of a variant before the next variant
#   * a new job starts only if it is expected to end before TIME_BUDGET_H
#     (expected duration = longest finished job so far, x1.1; 60 min before any finished)
#   * HARD_STOP_H: the notebook stops waiting and writes the tables regardless
#   * every finished job is saved immediately; the tables use whatever finished
#   Kaggle saves /kaggle/working ONLY if the session ends normally, so keep the
#   budget well below the GPU quota left on the account.
#
# Kaggle: GPU T4 x2, Internet ON (ResNet-50 ImageNet weights), inputs = BreaKHis
# images, binary artifacts dataset, binary code-files dataset, ZoomMoE results.
# Paste the "# %% [Cell N]" blocks into cells (or all into one cell), Save & Run All.
# =============================================================================


# %% [Cell 1] Imports and configuration
import dataclasses
import gc
import importlib.util
import os
import queue
import random
import sys
import threading
import time
import traceback
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms.v2.functional as TF
from PIL import Image
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode

warnings.filterwarnings("ignore")
T_SESSION = time.time()             # the budget counts from here (first cell)

SEED = 42
TASK = "binary"
ARTIFACT_DIR = Path("/kaggle/input/datasets/nabinkoirala452/articafts/artifacts")
CODE_DIR = Path("/kaggle/input/datasets/nabinkoirala452/code-files")
ZOOMMOE_FULL_CSV = Path("/kaggle/input/datasets/nabinkoirala452/zoomoe-results-binary-multiclass/zoommoe_results.csv")
# If the account cannot see the nabinkoirala452 datasets, add them via "Add Input" and
# fix the three paths above (the printed error will show which one is missing).

# priority order; each variant runs all 5 folds before the next one starts
ABLATIONS = {
    "no_moe":         dict(moe_stages=()),
    "moe_mag_only":   dict(cond_state_dim=0),
    "no_contrastive": dict(w_align=0.0, w_ordinal=0.0, w_disent=0.0, w_advmag=0.0, w_supcon=0.0),
    "full":           {},
    "no_supcon":      dict(w_supcon=0.0),
    "no_ordinal":     dict(w_ordinal=0.0),
}

TIME_BUDGET_H = 11.0    # no job may be expected to finish after this (hours since Cell 1)
HARD_STOP_H = 11.4      # stop waiting no matter what, write the tables (Kaggle kills sessions at 12 h)
STALL_MINUTES = 30      # no log line for this long -> stop waiting

BATCH = 8
EPOCHS_S0, EPOCHS_S1, EPOCHS_S2 = 8, 15, 8      # notebook 02
LR_S01, LR_S2 = 3e-4, 3e-5                      # notebook 02
SCALE_DROPOUT = 0.15                            # notebook 02
IMG_SIZE = 224
MAGS = (40, 100, 200, 400)

OUT = Path("/kaggle/working/zoommoe_ablation_binary")
JOBS_DIR = OUT / "jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)

N_GPUS = torch.cuda.device_count()
DEVICES = [torch.device(f"cuda:{i}") for i in range(N_GPUS)]
assert DEVICES, "no GPU found - switch the accelerator to GPU T4 x2"
print("torch", torch.__version__, "| GPUs:", [torch.cuda.get_device_name(i) for i in range(N_GPUS)])


def hours():
    return (time.time() - T_SESSION) / 3600


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# %% [Cell 2] Notebook-02 code, pooled samples, folds, images pre-loaded in RAM
def resolve(path, must_contain):
    """Kaggle mounts datasets at /kaggle/input/datasets/<owner>/<slug>/ (UI) or
    /kaggle/input/<slug>/ (older / API runs). Try both, then search by slug."""
    if (path / must_contain).exists():
        return path
    parts = path.parts
    i = parts.index("input")
    rest = parts[i + 1:]
    if rest and rest[0] == "datasets":
        rest = rest[2:]                       # drop "datasets/<owner>"
    slug, sub = rest[0], Path(*rest[1:]) if len(rest) > 1 else Path()
    for base in [Path("/kaggle/input") / slug, *Path("/kaggle/input").rglob(slug)]:
        if (base / sub / must_contain).exists():
            print(f"input {path} -> {base / sub}")
            return base / sub
    raise FileNotFoundError(f"missing input {path / must_contain}; inputs: {sorted(os.listdir('/kaggle/input'))}")


ARTIFACT_DIR = resolve(ARTIFACT_DIR, "samples_pool.pkl")
CODE_DIR = resolve(CODE_DIR, "zoommoe_model.py")
if not ZOOMMOE_FULL_CSV.exists():
    ZOOMMOE_FULL_CSV = resolve(ZOOMMOE_FULL_CSV.parent, ZOOMMOE_FULL_CSV.name) / ZOOMMOE_FULL_CSV.name
for p in (ARTIFACT_DIR / "folds_all.npy", ARTIFACT_DIR / "samples_all.pkl", CODE_DIR / "breakhis_common.py"):
    assert p.exists(), f"missing input: {p}"


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


bc = load_module(CODE_DIR / "breakhis_common.py", "bc_binary")

# The binary notebook 02 did NOT train with code-files/zoommoe_model.py: its cell 4 wrote its own
# copy with %%writefile. Verified (diff, 2026-10-21): the two differ ONLY in this line, the fp16
# autocast fix. The unpatched line crashes under AMP ("scatter(): Expected self.dtype to be equal
# to src.dtype"), so notebook 02's results necessarily came from the patched version.
_src = (CODE_DIR / "zoommoe_model.py").read_text()
_old = "gates = torch.zeros_like(logits).scatter(-1, topi, topv.softmax(dim=-1))"
_new = "gates = torch.zeros_like(logits).scatter(-1, topi, topv.softmax(dim=-1).to(logits.dtype))"
assert _src.count(_old) == 1, "code-files/zoommoe_model.py changed - re-verify against notebook 02"
_patched = Path("/kaggle/working/zoommoe_model_nb02.py")
_patched.write_text(_src.replace(_old, _new))
zm = load_module(_patched, "zm_binary")
print("zoommoe_model: code-files version + notebook-02 AMP fix (1 line)")
samples = pd.read_pickle(ARTIFACT_DIR / "samples_pool.pkl").reset_index(drop=True)
samples_ref = pd.read_pickle(ARTIFACT_DIR / "samples_all.pkl")
folds = list(np.load(ARTIFACT_DIR / "folds_all.npy", allow_pickle=True))
assert (samples.sample_id.values == samples_ref.sample_id.values).all(), "sample tables not aligned"
bc.audit_folds(samples, folds)

# exactly what binary notebook 02 printed for its folds (cell 9) -> same data, same splits
NB02_FOLDS = [  # (train, val, test) samples, (train, val, test) patients, test malignant fraction
    ((1638, 308, 461), (56, 10, 16), 0.753),
    ((1707, 309, 391), (56, 10, 16), 0.749),
    ((1622, 283, 502), (56, 10, 16), 0.793),
    ((1652, 233, 522), (55, 10, 17), 0.722),
    ((1608, 268, 531), (55, 10, 17), 0.501),
]
assert len(samples) == 2407 and samples.patient_id.nunique() == 82
for fi, (f, (n, pats, mal)) in enumerate(zip(folds, NB02_FOLDS)):
    got_n = tuple(len(f[k]) for k in ("train", "val", "test"))
    got_p = tuple(samples.iloc[f[k]].patient_id.nunique() for k in ("train", "val", "test"))
    got_m = round(float(samples.label.iloc[f["test"]].mean()), 3)
    assert (got_n, got_p, got_m) == (n, pats, mal), f"fold {fi} differs from notebook 02: {got_n} {got_p} {got_m}"
print("folds identical to binary notebook 02 (sizes, patients, test malignant fraction)")

_probe = next(p for lst in samples["cand40"] for p in lst)
if not Path(_probe).exists():
    print("remapping image paths by filename")
    _index = {p.name: str(p) for p in Path("/kaggle/input").rglob("SOB_*.png")}
    for m in MAGS:
        samples[f"cand{m}"] = samples[f"cand{m}"].apply(lambda lst: [_index[Path(p).name] for p in lst])

ALL_PATHS = sorted({p for m in MAGS for lst in samples[f"cand{m}"] for p in lst})
PATH2IDX = {p: i for i, p in enumerate(ALL_PATHS)}


def _load(p):
    # the Resize((224, 224)) breakhis_common applies first
    return np.asarray(Image.open(p).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR))


t0 = time.time()
with ThreadPoolExecutor(8) as ex:
    IMG_ARR = np.stack(list(ex.map(_load, ALL_PATHS)))
print(f"pre-loaded {IMG_ARR.shape} ({IMG_ARR.nbytes / 1e9:.2f} GB) in {time.time() - t0:.0f}s")


# %% [Cell 3] Dataset (breakhis_common sampling) + GPU augmentation, no worker processes
class MultiScaleRAM(Dataset):
    """breakhis_common.MultiScaleBreaKHis sampling (random candidate per magnification
    in training, cands[i % len] at eval, scale dropout) from RAM, raw uint8."""

    def __init__(self, samples, train, scale_dropout=0.0, seed=SEED):
        self.samples = samples.reset_index(drop=True)
        self.cands = [[[PATH2IDX[p] for p in row[f"cand{m}"]] for m in MAGS]
                      for _, row in self.samples.iterrows()]
        self.train, self.scale_dropout = train, scale_dropout
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        row = self.samples.iloc[i]
        imgs = torch.zeros(len(MAGS), IMG_SIZE, IMG_SIZE, 3, dtype=torch.uint8)
        mask = torch.zeros(len(MAGS), dtype=torch.bool)
        for j in range(len(MAGS)):
            c = self.cands[i][j]
            if not c:
                continue
            k = self.rng.choice(c) if self.train else c[i % len(c)]
            imgs[j] = torch.from_numpy(IMG_ARR[k])
            mask[j] = True
        loaded = mask.clone()
        if self.train and self.scale_dropout > 0:
            for j in range(len(MAGS)):
                if mask[j] and mask.sum() > 1 and self.rng.random() < self.scale_dropout:
                    mask[j] = False
        return {"images": imgs, "mask": mask, "loaded": loaded,
                "label": torch.tensor(float(row["label"])),            # binary: one logit, BCE
                "patient_idx": torch.tensor(int(row["patient_idx"])), "index": torch.tensor(int(i))}


def gpu_augment(x, rng):
    """breakhis_common train transform per image, on the GPU: flips, RandomRotation(90) p=0.5,
    ColorJitter(0.15, 0.15, 0.15, 0.03) in random order."""
    out = []
    for img in x:
        if rng.random() < 0.5:
            img = TF.horizontal_flip(img)
        if rng.random() < 0.5:
            img = TF.vertical_flip(img)
        if rng.random() < 0.5:
            img = TF.rotate(img, float(rng.uniform(-90, 90)), interpolation=InterpolationMode.NEAREST, fill=0.0)
        for op in rng.permutation(4):
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


class GPUBatches:
    """Worker-free loader: images -> GPU, augment (train), ImageNet-normalise, empty
    magnification slots back to exactly 0. Other fields stay on CPU (zm.collect expects that)."""

    def __init__(self, ds, batch_size, shuffle, drop_last, dev, seed):
        self.dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=0,
                             generator=torch.Generator().manual_seed(seed))
        self.dev, self.train = dev, ds.train
        self.mean = torch.tensor(bc.IMAGENET_MEAN, device=dev).view(1, 3, 1, 1)
        self.std = torch.tensor(bc.IMAGENET_STD, device=dev).view(1, 3, 1, 1)
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        for batch in self.dl:
            x = batch.pop("images")
            loaded = batch.pop("loaded")
            B, M = x.shape[:2]
            x = x.to(self.dev, non_blocking=True).reshape(B * M, IMG_SIZE, IMG_SIZE, 3)
            x = x.permute(0, 3, 1, 2).float().div_(255.0)
            if self.train:
                x = gpu_augment(x, self.rng)
            x = (x - self.mean) / self.std
            batch["images"] = x.reshape(B, M, 3, IMG_SIZE, IMG_SIZE) * loaded.to(self.dev).view(B, M, 1, 1, 1)
            yield batch


# %% [Cell 4] Build each variant once and run one real batch (fails fast, ~2 min)
BUILD_LOCK = threading.Lock()


def make_cfg(overrides):
    base = zm.ZoomMoEConfig(img_size=IMG_SIZE, n_experts=8, top_k=2, pretrained=True)
    return dataclasses.replace(base, **overrides)


def build(overrides):
    cfg = make_cfg(overrides)
    with BUILD_LOCK:      # ResNet-50 weight download must not race between threads
        model = zm.ZoomMoE(cfg)
    return model, cfg


print("smoke test:")
_dl = GPUBatches(MultiScaleRAM(samples.iloc[folds[0]["train"][:16]], True, SCALE_DROPOUT), 8, True, True,
                 DEVICES[0], SEED)
_batch = {k: v.to(DEVICES[0]) for k, v in next(iter(_dl)).items()}
for _name, _ov in ABLATIONS.items():
    _m, _cfg = build(_ov)
    _m = _m.to(DEVICES[0]).train()
    with torch.autocast("cuda", dtype=torch.float16):
        _out = _m(_batch["images"], _batch["mask"])
        _tot, _parts = zm.compute_losses(_out, _batch, _cfg)
    _tot.backward()
    assert torch.isfinite(_tot), _name
    _n = sum(p.numel() for p in _m.parameters() if p.requires_grad)
    if _name == "full":   # notebook 02 printed "trainable params: 48538906" for every fold
        assert _n == 48538906, f"full model has {_n} params, notebook 02 had 48538906"
    print(f"  {_name:15s} loss={float(_tot):.3f}  trainable params={_n:,}")
    del _m, _out, _tot
    torch.cuda.empty_cache()
del _dl, _batch
print("smoke test passed")


# %% [Cell 5] Training: notebook-02 run_stage, per GPU
LOCK_PRINT = threading.Lock()
LAST_PROGRESS = [time.time()]


def log(*a):
    with LOCK_PRINT:
        print(*a, flush=True)
        LAST_PROGRESS[0] = time.time()


def run_stage(model, loader, cfg, dev, epochs, lr, supervised, unsupervised, freeze, tag):
    model.backbone.freeze_until(freeze)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=max(epochs * len(loader), 1),
                                                pct_start=0.3)
    scaler = torch.amp.GradScaler("cuda")
    for ep in range(epochs):
        model.train()
        running, t0 = [], time.time()
        for it, batch in enumerate(loader):
            batch = {k: v.to(dev, non_blocking=True) for k, v in batch.items()}
            frac = (ep * len(loader) + it) / max(epochs * len(loader), 1)
            lam = float(2.0 / (1.0 + np.exp(-10 * frac)) - 1.0)
            with torch.autocast("cuda", dtype=torch.float16):
                out = model(batch["images"], batch["mask"], grl_lambda=lam)
                total, parts = zm.compute_losses(out, batch, cfg, supervised=supervised, unsupervised=unsupervised)
            opt.zero_grad(set_to_none=True)
            scaler.scale(total).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            running.append(parts)
        mean = {k: float(np.mean([r[k] for r in running])) for k in running[0]}
        log(f"  [{tag}] epoch {ep + 1}/{epochs} ({time.time() - t0:.0f}s) total={sum(mean.values()):.3f} "
            f"cls={mean.get('cls', float('nan')):.3f}")


def evaluate_loader(model, loader, dev):
    res = zm.collect(model, loader, dev)
    return bc.evaluate(res["label"], res["prob"], res["patient_idx"]), res


def job_file(name, fi):
    return JOBS_DIR / f"{name}__fold{fi}.npz"


def train_job(name, fi, dev):
    torch.cuda.set_device(dev)
    tag = f"{name}|f{fi}|{dev}"
    fold = folds[fi]
    set_seed(SEED + fi)                     # same seed per fold as notebook 02
    model, cfg = build(ABLATIONS[name])
    model = model.to(dev)

    def loader(idx, train):
        ds = MultiScaleRAM(samples.iloc[idx], train, SCALE_DROPOUT if train else 0.0, seed=SEED + fi)
        return GPUBatches(ds, BATCH if train else 32, train, train, dev, SEED + fi)

    tr, va, te = loader(fold["train"], True), loader(fold["val"], False), loader(fold["test"], False)
    t0 = time.time()
    log(f"[{tag}] start at {hours():.2f} h")
    run_stage(model, tr, cfg, dev, EPOCHS_S0, LR_S01, False, True, "layer2", f"{tag}-s0")
    run_stage(model, tr, cfg, dev, EPOCHS_S1, LR_S01, True, True, "layer2", f"{tag}-s1")
    run_stage(model, tr, cfg, dev, EPOCHS_S2, LR_S2, True, True, None, f"{tag}-s2")
    vm, val_res = evaluate_loader(model, va, dev)
    tm, test_res = evaluate_loader(model, te, dev)
    minutes = (time.time() - t0) / 60
    log(f"[{tag}] TEST: " + str({k: round(v, 4) for k, v in tm.items()}) + f"  ({minutes:.1f} min)")
    np.savez_compressed(job_file(name, fi), test_prob=test_res["prob"], test_label=test_res["label"],
                        test_pid=test_res["patient_idx"], val_prob=val_res["prob"], val_label=val_res["label"],
                        val_pid=val_res["patient_idx"], expert_usage=test_res["expert_usage"],
                        metrics=np.array([tm]), val_metrics=np.array([vm]), minutes=minutes)
    del model, tr, va, te
    gc.collect()
    torch.cuda.empty_cache()
    return minutes


# %% [Cell 6] Run the jobs in priority order on both GPUs, inside the time budget
jobs = queue.Queue()
for name in ABLATIONS:                       # variant-major = priority order
    for fi in range(len(folds)):
        if not job_file(name, fi).exists():
            jobs.put((name, fi))
print(f"{jobs.qsize()} jobs queued; budget {TIME_BUDGET_H} h (now {hours():.2f} h)")
DURATIONS, FAILED, SKIPPED = [], [], []
STATE_LOCK = threading.Lock()


def expected_minutes():
    with STATE_LOCK:
        return 1.1 * max(DURATIONS) if DURATIONS else 60.0


def gpu_worker(dev):
    while True:
        try:
            name, fi = jobs.get_nowait()
        except queue.Empty:
            return
        if hours() + expected_minutes() / 60 > TIME_BUDGET_H:
            with STATE_LOCK:
                SKIPPED.append((name, fi))
            continue                         # drain the queue without starting anything
        try:
            m = train_job(name, fi, dev)
            with STATE_LOCK:
                DURATIONS.append(m)
        except Exception as e:
            FAILED.append((name, fi, repr(e)))
            log(f"!! FAILED {name} fold {fi}: {e}\n{traceback.format_exc()}")
            gc.collect()
            torch.cuda.empty_cache()


LAST_PROGRESS[0] = time.time()
threads = [threading.Thread(target=gpu_worker, args=(d,), daemon=True) for d in DEVICES]
for th in threads:
    th.start()
while any(th.is_alive() for th in threads):
    for th in threads:
        th.join(timeout=30)
    if hours() > HARD_STOP_H:
        print(f"\n!! hard stop at {hours():.2f} h - writing tables with the finished jobs")
        break
    if time.time() - LAST_PROGRESS[0] > STALL_MINUTES * 60:
        print(f"\n!! no progress for {STALL_MINUTES} min - writing tables with the finished jobs")
        break
print(f"\nstopped at {hours():.2f} h | finished {len(DURATIONS)} jobs "
      f"(mean {np.mean(DURATIONS) if DURATIONS else 0:.1f} min) | skipped (budget) {len(SKIPPED)} | "
      f"failed {FAILED or 'none'}")


# %% [Cell 7] Ablation table (+ val-tuned threshold) vs notebook-02 ZoomMoE
def best_threshold(prob, label):
    """threshold maximising macro-F1 on the val split"""
    grid = np.linspace(0.05, 0.95, 91)
    f1s = [f1_score(label, (prob > t).astype(int), average="macro") for t in grid]
    return float(grid[int(np.argmax(f1s))])


rows = []
for name in ABLATIONS:
    for fi in range(len(folds)):
        f = job_file(name, fi)
        if not f.exists():
            continue
        z = np.load(f, allow_pickle=True)
        thr = best_threshold(z["val_prob"], z["val_label"])
        tuned = bc.evaluate(z["test_label"], z["test_prob"], z["test_pid"], threshold=thr)
        rows.append({"variant": name, "fold": fi, **z["metrics"][0], "minutes": float(z["minutes"]),
                     "val_thr": thr, **{f"{k}@thr": v for k, v in tuned.items()}})
res = pd.DataFrame(rows)
res.to_csv(OUT / "ablation_fold_results.csv", index=False)

if ZOOMMOE_FULL_CSV.exists():
    ref = pd.read_csv(ZOOMMOE_FULL_CSV)
    ref["variant"] = "full (notebook 02)"
    res_all = pd.concat([ref, res], ignore_index=True)
else:
    print("notebook-02 results not found:", ZOOMMOE_FULL_CSV)
    res_all = res

KEYS = ["acc", "prr", "f1_macro", "auc"]


def fmt(g):
    out = {k: f"{g[k].mean():.4f} ± {g[k].std():.4f}" for k in KEYS}
    out["folds"] = len(g)
    return pd.Series(out)


order = ["full (notebook 02)"] + list(ABLATIONS)
tab = res_all.groupby("variant").apply(fmt).reindex([v for v in order if v in set(res_all.variant)])
print("\n######## BINARY ablation — sample level, patient-disjoint folds ########")
display(tab)
tab.to_csv(OUT / "ablation_summary_binary.csv")

# paired difference vs notebook-02 full ZoomMoE, on the folds each variant finished
if ZOOMMOE_FULL_CSV.exists() and len(res):
    ref_i = ref.set_index("fold")
    deltas = []
    for name, g in res.groupby("variant", sort=False):
        g = g.set_index("fold")
        d = {f"Δ{k}": float((g[k] - ref_i.loc[g.index, k]).mean()) for k in KEYS}
        d.update({"variant": name, "folds": len(g)})
        deltas.append(d)
    dtab = pd.DataFrame(deltas).set_index("variant").round(4)
    print("\nvariant − full ZoomMoE (negative = removing the part hurts):")
    display(dtab)
    dtab.to_csv(OUT / "ablation_delta_vs_full.csv")

if len(res):
    thr_tab = res.groupby("variant")[[f"{k}@thr" for k in ["acc", "prr", "f1_macro"]] + ["val_thr"]].mean().round(4)
    print("\nsame models with the decision threshold tuned on the val split (macro-F1):")
    display(thr_tab)
    thr_tab.to_csv(OUT / "ablation_val_threshold.csv")
print(f"\ndone at {hours():.2f} h; outputs in {OUT}: {sorted(p.name for p in OUT.iterdir())}")
