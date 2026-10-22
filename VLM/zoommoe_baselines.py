# =============================================================================
# 09 - Same-backbone baselines: ResNet-50 trained EXACTLY like ZoomMoE, simple fusion
# =============================================================================
# Reviewer question: "Is ZoomMoE better than a ResNet-50 trained the same way with
# a simple multi-scale fusion?" (notebook 01's baselines use FROZEN VGG16 features.)
#
# Everything is notebook 02's ZoomMoE recipe: ImageNet ResNet-50 (IMAGENET1K_V2),
# pooled pairing (samples_pool.pkl), folds_all.npy, scale dropout 0.15, the
# breakhis_common augmentation, batch 8, stages 8/15/8 epochs with lr
# 3e-4/3e-4/3e-5, freeze up to layer2 in the first two stages, AdamW wd 1e-4,
# OneCycle (pct 0.3), grad clip 5, fp16, sample-level breakhis_common.evaluate.
# Only the model differs (no MoE, no progressive zoom, no auxiliary losses; the
# ZoomMoE stage 0 is unsupervised, here all stages are supervised, same epochs):
#
#   late_avg       shared ResNet-50 + head on every magnification image, loss on each
#                  image; prediction = mean probability over the sample's magnifications
#                  (= a single-image CNN, magnification-averaged)
#   mean_fusion    masked mean of the 4 pooled ResNet-50 features -> head
#   concat_fusion  the 4 pooled features concatenated (missing = 0) + mask bits -> head
#
# Binary: one logit + BCE (as ZoomMoE binary). 8-class: 8 logits + CE.
# Both tasks; 8-class first. Two GPUs, one job per GPU, time-budget guard.
# Kaggle: GPU T4 x2, Internet ON.
# =============================================================================


# %% [Cell 1] Imports and configuration
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
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.v2.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torchvision.transforms import InterpolationMode

warnings.filterwarnings("ignore")
T_SESSION = time.time()

SEED = 42
VARIANTS = ["mean_fusion", "late_avg", "concat_fusion"]      # priority order
TASKS = ["multiclass", "binary"]                              # 8-class first

INPUTS = {
    "binary": dict(code=Path("/kaggle/input/datasets/nabinkoirala452/code-files"),
                   art=Path("/kaggle/input/datasets/nabinkoirala452/articafts/artifacts")),
    "multiclass": dict(code=Path("/kaggle/input/datasets/nabinkoirala452/code-files-mc"),
                       art=Path("/kaggle/input/datasets/nabinkoirala452/articafts-notebook1-mc/artifacts")),
}
ZOOMMOE_RESULTS = Path("/kaggle/input/datasets/nabinkoirala452/zoomoe-results-binary-multiclass")

# printed by notebook 02 (cell 9) for each task: (train, val, test) sizes, (train, val, test) patients
NB02_FOLDS = {
    "binary": [((1638, 308, 461), (56, 10, 16)), ((1707, 309, 391), (56, 10, 16)), ((1622, 283, 502), (56, 10, 16)),
               ((1652, 233, 522), (55, 10, 17)), ((1608, 268, 531), (55, 10, 17))],
    "multiclass": [((1559, 324, 524), (56, 10, 16)), ((1616, 287, 504), (56, 10, 16)), ((1703, 228, 476), (55, 10, 17)),
                   ((1625, 302, 480), (56, 10, 16)), ((1701, 283, 423), (55, 10, 17))],
}
# 8-class test class counts printed by notebook 02 (strongest fold check for 8-class)
NB02_MC_TEST_COUNTS = [[0, 60, 66, 56, 248, 55, 22, 17], [0, 56, 0, 0, 298, 55, 57, 38],
                       [32, 81, 42, 37, 196, 0, 43, 45], [71, 121, 17, 31, 149, 56, 0, 35],
                       [17, 0, 0, 39, 181, 35, 132, 19]]

TIME_BUDGET_H = 11.0
HARD_STOP_H = 11.4
STALL_MINUTES = 30

BATCH = 8
EPOCHS = (8, 15, 8)
LRS = (3e-4, 3e-4, 3e-5)
FREEZE = ("layer2", "layer2", None)
SCALE_DROPOUT = 0.15
IMG_SIZE = 224
MAGS = (40, 100, 200, 400)
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

OUT = Path("/kaggle/working/baselines_resnet50")
JOBS_DIR = OUT / "jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)
N_GPUS = torch.cuda.device_count()
DEVICES = [torch.device(f"cuda:{i}") for i in range(N_GPUS)]
assert DEVICES, "no GPU"
print("torch", torch.__version__, "| GPUs:", [torch.cuda.get_device_name(i) for i in range(N_GPUS)])


def hours():
    return (time.time() - T_SESSION) / 3600


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve(path, must_contain):
    """/kaggle/input/datasets/<owner>/<slug>/... (UI) or /kaggle/input/<slug>/... (API)."""
    if (path / must_contain).exists():
        return path
    parts = path.parts
    rest = parts[parts.index("input") + 1:]
    if rest and rest[0] == "datasets":
        rest = rest[2:]
    slug, sub = rest[0], Path(*rest[1:]) if len(rest) > 1 else Path()
    for base in [Path("/kaggle/input") / slug, *Path("/kaggle/input").rglob(slug)]:
        if (base / sub / must_contain).exists():
            return base / sub
    raise FileNotFoundError(f"missing {path / must_contain}; inputs: {sorted(os.listdir('/kaggle/input'))}")


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# %% [Cell 2] Samples, folds (checked against notebook 02), images in RAM
TASK = {}
for task, inp in INPUTS.items():
    code = resolve(inp["code"], "breakhis_common.py")
    art = resolve(inp["art"], "samples_pool.pkl")
    bc = load_module(code / "breakhis_common.py", f"bc_{task}")
    samples = pd.read_pickle(art / "samples_pool.pkl").reset_index(drop=True)
    ref = pd.read_pickle(art / "samples_all.pkl")
    folds = list(np.load(art / "folds_all.npy", allow_pickle=True))
    assert (samples.sample_id.values == ref.sample_id.values).all()
    for fi, (f, (n, p)) in enumerate(zip(folds, NB02_FOLDS[task])):
        got_n = tuple(len(f[k]) for k in ("train", "val", "test"))
        got_p = tuple(samples.iloc[f[k]].patient_id.nunique() for k in ("train", "val", "test"))
        assert (got_n, got_p) == (n, p), f"{task} fold {fi} differs from notebook 02: {got_n} {got_p}"
        if task == "multiclass":
            cnt = samples.label.iloc[f["test"]].value_counts().reindex(range(8), fill_value=0).tolist()
            assert cnt == NB02_MC_TEST_COUNTS[fi], f"multiclass fold {fi} test classes differ: {cnt}"
    TASK[task] = dict(bc=bc, samples=samples, folds=folds, n_out=getattr(bc, "N_CLASSES", 1))
    print(f"[{task}] {len(samples)} samples, {samples.patient_id.nunique()} patients, n_out={TASK[task]['n_out']}; "
          f"folds identical to notebook 02")

_probe = next(p for lst in TASK["binary"]["samples"]["cand40"] for p in lst)
if not Path(_probe).exists():
    print("remapping image paths by filename")
    _index = {p.name: str(p) for p in Path("/kaggle/input").rglob("SOB_*.png")}
    for t in TASK.values():
        for m in MAGS:
            t["samples"][f"cand{m}"] = t["samples"][f"cand{m}"].apply(lambda lst: [_index[Path(p).name] for p in lst])

ALL_PATHS = sorted({p for t in TASK.values() for m in MAGS for lst in t["samples"][f"cand{m}"] for p in lst})
PATH2IDX = {p: i for i, p in enumerate(ALL_PATHS)}
t0 = time.time()
with ThreadPoolExecutor(8) as ex:
    IMG_ARR = np.stack(list(ex.map(
        lambda p: np.asarray(Image.open(p).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)), ALL_PATHS)))
print(f"pre-loaded {IMG_ARR.shape} ({IMG_ARR.nbytes / 1e9:.2f} GB) in {time.time() - t0:.0f}s")


# %% [Cell 3] Dataset (breakhis_common sampling) + GPU augmentation (no worker processes)
class MultiScaleRAM(Dataset):
    def __init__(self, samples, train, binary, scale_dropout=0.0, seed=SEED):
        self.samples = samples.reset_index(drop=True)
        self.cands = [[[PATH2IDX[p] for p in row[f"cand{m}"]] for m in MAGS] for _, row in self.samples.iterrows()]
        self.train, self.binary, self.scale_dropout = train, binary, scale_dropout
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
        label = torch.tensor(float(row["label"])) if self.binary else torch.tensor(int(row["label"]), dtype=torch.long)
        return {"images": imgs, "mask": mask, "loaded": loaded, "label": label,
                "patient_idx": torch.tensor(int(row["patient_idx"]))}


def gpu_augment(x, rng):
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
    def __init__(self, ds, batch_size, shuffle, drop_last, dev, seed):
        self.dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=0,
                             generator=torch.Generator().manual_seed(seed))
        self.dev, self.train = dev, ds.train
        self.mean = torch.tensor(IMAGENET_MEAN, device=dev).view(1, 3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, device=dev).view(1, 3, 1, 1)
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        for batch in self.dl:
            x = batch.pop("images")
            loaded = batch.pop("loaded")
            B, M = x.shape[:2]
            x = x.to(self.dev, non_blocking=True).reshape(B * M, IMG_SIZE, IMG_SIZE, 3).permute(0, 3, 1, 2).float().div_(255.0)
            if self.train:
                x = gpu_augment(x, self.rng)
            x = (x - self.mean) / self.std
            batch["images"] = x.reshape(B, M, 3, IMG_SIZE, IMG_SIZE) * loaded.to(self.dev).view(B, M, 1, 1, 1)
            yield batch


# %% [Cell 4] Baseline models (same ResNet-50 and freeze_until as ZoomMoE's MoEBackbone)
class ResNetBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        r = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = r.layer1, r.layer2, r.layer3, r.layer4

    def freeze_until(self, stage):           # identical to zoommoe_model.MoEBackbone.freeze_until
        order = ["stem", "layer1", "layer2", "layer3", "layer4"]
        if stage is None:
            for p in self.parameters():
                p.requires_grad = True
            return
        stop = order.index(stage)
        for i, name in enumerate(order):
            for p in getattr(self, name).parameters():
                p.requires_grad = i > stop

    def forward(self, x):
        for m in (self.stem, self.layer1, self.layer2, self.layer3, self.layer4):
            x = m(x)
        return x.mean(dim=(2, 3))            # B, 2048


class FusionBaseline(nn.Module):
    def __init__(self, mode, n_out, dropout=0.3):
        super().__init__()
        self.mode, self.n_out = mode, n_out
        self.backbone = ResNetBackbone()
        d = 2048
        in_dim = {"late_avg": d, "mean_fusion": d, "concat_fusion": 4 * d + 4}[mode]
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_dim, n_out))

    def forward(self, images, mask):
        B, M = images.shape[:2]
        f = self.backbone(images.flatten(0, 1)).view(B, M, -1)          # B, M, 2048
        mk = mask.float().unsqueeze(-1)
        if self.mode == "late_avg":
            return self.head(f)                                         # B, M, n_out (per image)
        if self.mode == "mean_fusion":
            return self.head((f * mk).sum(1) / mk.sum(1).clamp(min=1))
        return self.head(torch.cat([(f * mk).flatten(1), mask.float()], dim=1))

    def loss(self, out, y, mask):
        if self.mode == "late_avg":                                     # every visible image gets the sample label
            B, M = out.shape[:2]
            sel = mask.reshape(-1)
            o = out.reshape(B * M, -1)[sel]
            yy = y.unsqueeze(1).expand(-1, M).reshape(B * M)[sel]
            return F.binary_cross_entropy_with_logits(o.squeeze(-1), yy) if self.n_out == 1 else F.cross_entropy(o, yy)
        return F.binary_cross_entropy_with_logits(out.squeeze(-1), y) if self.n_out == 1 else F.cross_entropy(out, y)

    @torch.no_grad()
    def predict(self, images, mask):
        out = self.forward(images, mask).float()
        p = torch.sigmoid(out.squeeze(-1)) if self.n_out == 1 else torch.softmax(out, -1)
        if self.mode == "late_avg":                                     # mean probability over visible images
            mk = mask.float() if self.n_out == 1 else mask.float().unsqueeze(-1)
            p = (p * mk).sum(1) / mk.sum(1).clamp(min=1)
        return p


BUILD_LOCK = threading.Lock()


def build(mode, task):
    with BUILD_LOCK:
        return FusionBaseline(mode, TASK[task]["n_out"])


# smoke test: every variant x task, one real batch forward/backward + predict
print("smoke test:")
for _task in TASKS:
    _s = TASK[_task]["samples"]
    _dl = GPUBatches(MultiScaleRAM(_s.iloc[TASK[_task]["folds"][0]["train"][:16]], True, _task == "binary",
                                   SCALE_DROPOUT), 8, True, True, DEVICES[0], SEED)
    _b = {k: v.to(DEVICES[0]) for k, v in next(iter(_dl)).items()}
    for _v in VARIANTS:
        _m = build(_v, _task).to(DEVICES[0]).train()
        with torch.autocast("cuda", dtype=torch.float16):
            _l = _m.loss(_m(_b["images"], _b["mask"]), _b["label"], _b["mask"])
        _l.backward()
        _p = _m.eval().predict(_b["images"], _b["mask"])
        assert torch.isfinite(_l) and _p.shape[0] == 8
        print(f"  {_task:10s} {_v:14s} loss={float(_l):.3f} pred{tuple(_p.shape)} "
              f"params={sum(p.numel() for p in _m.parameters()) / 1e6:.2f}M")
        del _m
        torch.cuda.empty_cache()
print("smoke test passed")


# %% [Cell 5] Training (ZoomMoE's run_stage with a plain supervised loss) and evaluation
LOCK_PRINT = threading.Lock()
LAST_PROGRESS = [time.time()]


def log(*a):
    with LOCK_PRINT:
        print(*a, flush=True)
        LAST_PROGRESS[0] = time.time()


def run_stage(model, loader, dev, epochs, lr, freeze, tag):
    model.backbone.freeze_until(freeze)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=max(epochs * len(loader), 1), pct_start=0.3)
    scaler = torch.amp.GradScaler("cuda")
    for ep in range(epochs):
        model.train()
        losses, t0 = [], time.time()
        for batch in loader:
            batch = {k: v.to(dev, non_blocking=True) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model.loss(model(batch["images"], batch["mask"]), batch["label"], batch["mask"])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(loss.item())
        log(f"  [{tag}] epoch {ep + 1}/{epochs} ({time.time() - t0:.0f}s) loss={np.mean(losses):.4f}")


@torch.no_grad()
def collect(model, loader, dev):
    model.eval()
    probs, ys, pids = [], [], []
    for batch in loader:
        with torch.autocast("cuda", dtype=torch.float16):
            p = model.predict(batch["images"].to(dev), batch["mask"].to(dev))
        probs.append(p.float().cpu().numpy())
        ys.append(batch["label"].numpy())
        pids.append(batch["patient_idx"].numpy())
    return np.concatenate(probs), np.concatenate(ys), np.concatenate(pids)


def job_file(v, task, fi):
    return JOBS_DIR / f"{v}__{task}__fold{fi}.npz"


def train_job(v, task, fi, dev):
    torch.cuda.set_device(dev)
    tag = f"{v}|{task}|f{fi}|{dev}"
    t, fold = TASK[task], TASK[task]["folds"][fi]
    set_seed(SEED + fi)
    model = build(v, task).to(dev)

    def loader(idx, train):
        ds = MultiScaleRAM(t["samples"].iloc[idx], train, task == "binary", SCALE_DROPOUT if train else 0.0, SEED + fi)
        return GPUBatches(ds, BATCH if train else 32, train, train, dev, SEED + fi)

    tr, va, te = loader(fold["train"], True), loader(fold["val"], False), loader(fold["test"], False)
    t0 = time.time()
    for s, (ep, lr, fr) in enumerate(zip(EPOCHS, LRS, FREEZE)):
        run_stage(model, tr, dev, ep, lr, fr, f"{tag}-s{s}")
    vp, vy, vpid = collect(model, va, dev)
    tp, ty, tpid = collect(model, te, dev)
    tm = t["bc"].evaluate(ty, tp, tpid)
    minutes = (time.time() - t0) / 60
    log(f"[{tag}] TEST: " + str({k: round(x, 4) for k, x in tm.items()}) + f"  ({minutes:.1f} min, at {hours():.2f} h)")
    np.savez_compressed(job_file(v, task, fi), test_prob=tp, test_label=ty, test_pid=tpid, val_prob=vp,
                        val_label=vy, val_pid=vpid, metrics=np.array([tm]), minutes=minutes)
    del model, tr, va, te
    gc.collect()
    torch.cuda.empty_cache()
    return minutes


# %% [Cell 6] Run all jobs (priority order) on both GPUs inside the time budget
jobs = queue.Queue()
for v in VARIANTS:
    for task in TASKS:
        for fi in range(5):
            if not job_file(v, task, fi).exists():
                jobs.put((v, task, fi))
print(f"{jobs.qsize()} jobs; budget {TIME_BUDGET_H} h (now {hours():.2f} h)")
DURATIONS, FAILED, SKIPPED = [], [], []
STATE = threading.Lock()


def gpu_worker(dev):
    while True:
        try:
            v, task, fi = jobs.get_nowait()
        except queue.Empty:
            return
        with STATE:
            exp = 1.1 * max(DURATIONS) if DURATIONS else 60.0
        if hours() + exp / 60 > TIME_BUDGET_H:
            SKIPPED.append((v, task, fi))
            continue
        try:
            m = train_job(v, task, fi, dev)
            with STATE:
                DURATIONS.append(m)
        except Exception as e:
            FAILED.append((v, task, fi, repr(e)))
            log(f"!! FAILED {v} {task} fold {fi}: {e}\n{traceback.format_exc()}")
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
        print("\n!! hard stop - writing tables with the finished jobs")
        break
    if time.time() - LAST_PROGRESS[0] > STALL_MINUTES * 60:
        print(f"\n!! no progress for {STALL_MINUTES} min - writing tables with the finished jobs")
        break
print(f"\nstopped at {hours():.2f} h | finished {len(DURATIONS)} | skipped {len(SKIPPED)} | failed {FAILED or 'none'}")


# %% [Cell 7] Tables: baselines vs ZoomMoE (notebook 02), same folds and evaluation
rows = []
for v in VARIANTS:
    for task in TASKS:
        for fi in range(5):
            f = job_file(v, task, fi)
            if f.exists():
                z = np.load(f, allow_pickle=True)
                rows.append({"variant": f"ResNet-50 {v}", "task": task, "fold": fi, **z["metrics"][0],
                             "minutes": float(z["minutes"])})
res = pd.DataFrame(rows)
res.to_csv(OUT / "baselines_fold_results.csv", index=False)

zr = resolve(ZOOMMOE_RESULTS, "zoommoe_results.csv")
for task, fn in [("binary", "zoommoe_results.csv"), ("multiclass", "zoommoe_multiclass_results.csv")]:
    z = pd.read_csv(zr / fn)
    z["variant"], z["task"] = "ZoomMoE + ResNet-50 (notebook 02)", task
    res = pd.concat([res, z], ignore_index=True)

K = ["acc", "prr", "f1_macro", "auc"]
for task in TASKS:
    g = res[res.task == task]
    tab = g.groupby("variant", sort=False).apply(
        lambda d: pd.Series({**{k: f"{d[k].mean():.4f} ± {d[k].std():.4f}" for k in K}, "folds": len(d)}))
    print(f"\n######## {task.upper()} ########")
    display(tab)
    tab.to_csv(OUT / f"baselines_summary_{task}.csv")
    ref = g[g.variant.str.startswith("ZoomMoE")].set_index("fold")
    for v, d in g[~g.variant.str.startswith("ZoomMoE")].groupby("variant", sort=False):
        d = d.set_index("fold")
        print(f"  ZoomMoE − {v}: " + " | ".join(
            f"{k} {(ref.loc[d.index, k] - d[k]).mean():+.4f} (ZoomMoE wins {int((ref.loc[d.index, k] > d[k]).sum())}/{len(d)})"
            for k in K))
print(f"\ndone at {hours():.2f} h")
