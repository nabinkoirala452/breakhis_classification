# =============================================================================
# 06 - ZoomMoE on VLM image encoders  (binary + 8-class)
# =============================================================================
# The ZoomMoE method is kept EXACTLY as in notebook 02 - same zoommoe_model.py
# (progressive zoom refinement, magnification/state-conditioned routing, all
# losses, heads), same 3-stage schedule, same pooled-pairing dataset, same
# patient-disjoint folds, same breakhis_common.evaluate. The only change is the
# backbone:
#
#   notebook 02 : ResNet-50 (ImageNet)  + Conv-MoE after layer3 and layer4
#   this file   : VLM ViT image encoder + Token-MoE after block 2L/3 and block L
#                 (QuiltNet / PLIP = ViT-B/32 -> 7x7 patch grid, the same 7x7
#                  feature map ResNet-50 gives at 224 px, so every downstream
#                  ZoomMoE module is reused unchanged)
#
# Two protocol details differ, both forced by the backbone:
#   * images are normalised with the VLM's own mean/std (not ImageNet's)
#   * pretrained ViT weights get 0.1x the stage learning rate (3e-4 would wreck
#     a CLIP encoder); MoE blocks and all ZoomMoE heads use the original lr.
#
# Kaggle setup: GPU T4 x2, Internet ON, inputs =
#   BreaKHis images, notebook-01 artifact datasets (samples_pool.pkl,
#   samples_all.pkl, folds_all.npy) and the code-files datasets
#   (breakhis_common.py + zoommoe_model.py, binary and multiclass versions).
#
# Paste each "# %% [Cell N]" block into its own cell, or everything in one cell.
# Every (backbone, task, fold) is a job; the two T4s run two jobs at a time and
# each finished job is saved, so a re-run in the same session skips it.
# =============================================================================


# %% [Cell 1] Install pinned libraries (same versions as the other VLM runs)
import subprocess
import sys

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q",
     "transformers==4.51.3", "accelerate>=0.34", "open_clip_torch==2.32.0"],
    check=True,
)


# %% [Cell 2] Imports and configuration
import dataclasses
import gc
import importlib.util
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
import torchvision.transforms.v2.functional as TF
from torch.utils.data import DataLoader, Dataset
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
CODE_DIRS = {   # breakhis_common.py + zoommoe_model.py of each task
    "binary": Path("/kaggle/input/datasets/nabinkoirala452/code-files"),
    "multiclass": Path("/kaggle/input/datasets/nabinkoirala452/code-files-mc"),
}
ZOOMMOE_RESULTS = {   # ZoomMoE + ResNet-50 (notebook 02), for the comparison table
    "binary": Path("/kaggle/input/datasets/nabinkoirala452/zoomoe-results-binary-multiclass/zoommoe_results.csv"),
    "multiclass": Path("/kaggle/input/datasets/nabinkoirala452/zoomoe-results-binary-multiclass/zoommoe_multiclass_results.csv"),
}
# optional: all_results_fold_level.csv from the VLM fine-tuning run (zero-shot,
# linear probe, fine-tuned rows), uploaded as a dataset
PREV_RESULTS = None   # e.g. Path("/kaggle/input/<vlm-ft-output>/all_results_fold_level.csv")

# backbone name -> (library, hub id)
BACKBONES = {
    "QuiltNet B/32": ("open_clip", "hf-hub:wisdomik/QuiltNet-B-32"),
    "PLIP":          ("hf_clip", "vinid/plip"),
    # ViT-B/16 (14x14 tokens, pooled to 7x7), ~2-3x slower; enable if time allows
    # "BiomedCLIP":  ("open_clip", "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"),
}

QUICK = False           # True -> 1/8 of the samples and epochs (1, 2, 1): pipeline check

BATCH = 8               # same as notebook 02 (4 backbone passes per sample)
EPOCHS_S0, EPOCHS_S1, EPOCHS_S2 = (1, 2, 1) if QUICK else (8, 15, 8)   # same as notebook 02
LR_S01, LR_S2 = 3e-4, 3e-5                                            # same as notebook 02
VIT_LR_MULT = 0.1       # pretrained ViT weights train at 0.1x the stage lr
SCALE_DROPOUT = 0.15    # same as notebook 02
IMG_SIZE = 224
STALL_MINUTES = 30      # no log line from any job for this long -> stop waiting, build tables anyway
SAVE_MODELS = True      # final checkpoint per (backbone, task, fold), fp16, ~190 MB each

OUT = Path("/kaggle/working/zoommoe_vlm_results_quick" if QUICK else "/kaggle/working/zoommoe_vlm_results")
JOBS_DIR = OUT / "jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR = OUT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

N_GPUS = torch.cuda.device_count()
DEVICES = [torch.device(f"cuda:{i}") for i in range(N_GPUS)]
assert DEVICES, "no GPU found - switch the accelerator to GPU T4 x2"
MAGS = (40, 100, 200, 400)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_name(name):
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")


import transformers, open_clip  # noqa: E402
print("torch", torch.__version__, "| transformers", transformers.__version__, "| open_clip", open_clip.__version__)
print("GPUs:", [torch.cuda.get_device_name(i) for i in range(N_GPUS)])


# %% [Cell 3] Task code (bc, zm), pooled samples, folds, images pre-loaded in RAM
def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


TASK = {}
for t in TASKS:
    bc = load_module(CODE_DIRS[t] / "breakhis_common.py", f"bc_{t}")
    zm = load_module(CODE_DIRS[t] / "zoommoe_model.py", f"zm_{t}")
    art = ARTIFACT_DIRS[t]
    samples = pd.read_pickle(art / "samples_pool.pkl")       # pooled pairing, as notebook 02
    samples_ref = pd.read_pickle(art / "samples_all.pkl")
    folds = list(np.load(art / "folds_all.npy", allow_pickle=True))
    assert (samples.sample_id.values == samples_ref.sample_id.values).all(), "sample tables not aligned"
    bc.audit_folds(samples, folds)
    TASK[t] = {"bc": bc, "zm": zm, "samples": samples.reset_index(drop=True), "folds": folds,
               "n_classes": getattr(bc, "N_CLASSES", 1)}   # binary model has one logit
print({t: TASK[t]["n_classes"] for t in TASKS}, "outputs per task")

# remap image paths if BreaKHis is mounted somewhere else
_probe = next(p for lst in TASK["binary"]["samples"]["cand40"] for p in lst)
if not Path(_probe).exists():
    print("remapping image paths by filename")
    _index = {p.name: str(p) for p in Path("/kaggle/input").rglob("SOB_*.png")}
    for t in TASKS:
        for m in MAGS:
            TASK[t]["samples"][f"cand{m}"] = TASK[t]["samples"][f"cand{m}"].apply(
                lambda lst: [_index[Path(p).name] for p in lst])

if QUICK:
    for t in TASKS:
        s = TASK[t]["samples"]
        keep = (s.groupby("patient_id").cumcount() % 8 == 0).values
        new_idx = -np.ones(len(s), dtype=int)
        new_idx[keep] = np.arange(keep.sum())
        TASK[t]["samples"] = s[keep].reset_index(drop=True)
        TASK[t]["folds"] = [{k: new_idx[v][new_idx[v] >= 0] for k, v in f.items()} for f in TASK[t]["folds"]]

for t in TASKS:
    print(f"[{t}] samples={len(TASK[t]['samples'])} "
          f"fold sizes={[{k: len(v) for k, v in f.items()} for f in TASK[t]['folds']][:1]} ...")

ALL_PATHS = sorted({p for t in TASKS for m in MAGS for lst in TASK[t]["samples"][f"cand{m}"] for p in lst})
PATH2IDX = {p: i for i, p in enumerate(ALL_PATHS)}


def _load(p):
    # same Resize((224, 224)) that breakhis_common applies first
    return np.asarray(Image.open(p).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR))


t0 = time.time()
with ThreadPoolExecutor(8) as ex:
    IMG_ARR = np.stack(list(ex.map(_load, ALL_PATHS)))
print(f"pre-loaded {IMG_ARR.shape} ({IMG_ARR.nbytes / 1e9:.2f} GB) in {time.time() - t0:.0f}s")


# %% [Cell 4] Dataset: breakhis_common.make_image_dataset, reading from RAM
class MultiScaleRAM(Dataset):
    """Same sampling as breakhis_common.MultiScaleBreaKHis (random candidate per
    magnification while training, cands[i % len] at eval, scale dropout), but
    images come from IMG_ARR as raw uint8. Augmentation + normalisation happen
    on the GPU in GPUBatches (no DataLoader worker processes: forking them from
    two GPU threads can deadlock, which hung a GPU in an earlier run)."""

    def __init__(self, samples, train, binary, scale_dropout=0.0, seed=SEED):
        self.samples = samples.reset_index(drop=True)
        self.cands = [[[PATH2IDX[p] for p in row[f"cand{m}"]] for m in MAGS]
                      for _, row in self.samples.iterrows()]
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
        loaded = mask.clone()          # slots with an image (scale dropout only hides them)
        if self.train and self.scale_dropout > 0:
            for j in range(len(MAGS)):
                if mask[j] and mask.sum() > 1 and self.rng.random() < self.scale_dropout:
                    mask[j] = False
        label = torch.tensor(float(row["label"])) if self.binary else torch.tensor(int(row["label"]), dtype=torch.long)
        return {"images": imgs, "mask": mask, "loaded": loaded, "label": label,
                "patient_idx": torch.tensor(int(row["patient_idx"])), "index": torch.tensor(int(i))}


def gpu_augment(x, rng):
    """
    x: N x 3 x H x W float in [0, 1] on the GPU. Per image, the breakhis_common
    train transform: RandomHorizontalFlip, RandomVerticalFlip,
    RandomApply([RandomRotation(90)], p=0.5), ColorJitter(0.15, 0.15, 0.15, 0.03).
    """
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


class GPUBatches:
    """
    Wraps a worker-free DataLoader: images go to `dev`, are augmented (train),
    normalised with the backbone mean/std, and empty magnification slots are set
    back to exactly 0 (as in breakhis_common). Other fields stay on the CPU, which
    is what zoommoe_model.collect expects.
    """

    def __init__(self, ds, batch_size, shuffle, drop_last, dev, mean, std, seed):
        self.dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=0,
                             generator=torch.Generator().manual_seed(seed))
        self.dev, self.train = dev, ds.train
        self.mean = torch.tensor(mean, device=dev).view(1, 3, 1, 1)
        self.std = torch.tensor(std, device=dev).view(1, 3, 1, 1)
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        for batch in self.dl:
            x = batch.pop("images")                                  # B, M, H, W, 3 uint8
            loaded = batch.pop("loaded")
            B, M = x.shape[:2]
            x = x.to(self.dev, non_blocking=True).reshape(B * M, IMG_SIZE, IMG_SIZE, 3)
            x = x.permute(0, 3, 1, 2).float().div_(255.0)
            if self.train:
                x = gpu_augment(x, self.rng)
            x = (x - self.mean) / self.std
            x = x.reshape(B, M, 3, IMG_SIZE, IMG_SIZE) * loaded.to(self.dev).view(B, M, 1, 1, 1)
            batch["images"] = x
            yield batch


# %% [Cell 5] VLM ViT backbone with zoom-conditioned Token-MoE (drop-in for MoEBackbone)
class TokenMoE(nn.Module):
    """
    Transformer counterpart of ConvMoE: sparse top-k mixture of MLP experts with
    per-token routing on [token ; magnification embedding ; zoom state h_{k-1}].
    Expert output layers start at zero, so at initialisation the block is the
    identity and the pretrained VLM features are untouched.
    Returns the same routing statistics as ConvMoE (gates, frac, zloss).
    """

    def __init__(self, dim, cond_dim, n_experts=8, top_k=2, ratio=4):
        super().__init__()
        self.n_experts, self.top_k = n_experts, top_k
        hidden = max(dim // ratio, 32)
        self.norm = nn.LayerNorm(dim)
        self.router = nn.Linear(dim + cond_dim, n_experts)
        self.experts = nn.ModuleList(
            [nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim)) for _ in range(n_experts)]
        )
        nn.init.zeros_(self.router.bias)
        for e in self.experts:
            nn.init.zeros_(e[-1].weight)
            nn.init.zeros_(e[-1].bias)

    def forward(self, x, cond):                     # x: B, N, C
        B, N, _ = x.shape
        u = self.norm(x)
        c = cond.unsqueeze(1).expand(-1, N, -1).to(u.dtype)
        logits = self.router(torch.cat([u, c], dim=-1))
        if self.training:
            logits = logits + torch.randn_like(logits) / self.n_experts
        topv, topi = logits.topk(self.top_k, dim=-1)
        gates = torch.zeros_like(logits).scatter(-1, topi, topv.softmax(dim=-1).to(logits.dtype))
        out = x
        for e, expert in enumerate(self.experts):
            g = gates[..., e]
            if not self.training and float(g.detach().abs().sum()) == 0.0:
                continue
            out = out + g.unsqueeze(-1).to(x.dtype) * expert(u).to(x.dtype)
        stats = {"gates": gates.mean(dim=(0, 1)), "frac": (gates > 0).float().mean(dim=(0, 1)),
                 "zloss": logits.logsumexp(dim=-1).pow(2).mean()}
        return out, stats


class _ViTAdapter(nn.Module):
    """Uniform view of a ViT: embed() -> tokens (B, P+N, C), blocks, final norm."""
    n_prefix = 1

    def run_block(self, blk, x):
        return blk(x)


class HFCLIPAdapter(_ViTAdapter):          # transformers CLIPVisionTransformer (PLIP, CLIP)
    def __init__(self, vm):
        super().__init__()
        self.vm = vm
        self.blocks = vm.encoder.layers
        self.final_norm = vm.post_layernorm
        self.width = vm.config.hidden_size
        self.embed_modules = [vm.embeddings, vm.pre_layrnorm]

    def embed(self, x):
        return self.vm.pre_layrnorm(self.vm.embeddings(x))

    def run_block(self, blk, x):
        return blk(x, None, None)[0]


class OpenCLIPAdapter(_ViTAdapter):        # open_clip VisionTransformer (QuiltNet, OpenAI-style ViTs)
    def __init__(self, v):
        super().__init__()
        self.v = v
        self.blocks = v.transformer.resblocks
        self.final_norm = v.ln_post
        self.width = v.transformer.width
        self.batch_first = getattr(v.transformer, "batch_first", True)
        self.embed_modules = [v.conv1, v.ln_pre]
        self.embed_params = [v.class_embedding, v.positional_embedding]

    def embed(self, x):
        v = self.v
        x = v.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        cls = v.class_embedding.view(1, 1, -1).expand(x.shape[0], -1, -1).to(x.dtype)
        x = torch.cat([cls, x], dim=1) + v.positional_embedding.to(x.dtype)
        x = v.patch_dropout(x)
        return v.ln_pre(x)

    def run_block(self, blk, x):
        if self.batch_first:
            return blk(x)
        return blk(x.transpose(0, 1)).transpose(0, 1)


class TimmAdapter(_ViTAdapter):            # timm VisionTransformer (BiomedCLIP trunk)
    def __init__(self, trunk):
        super().__init__()
        self.trunk = trunk
        self.blocks = trunk.blocks
        self.final_norm = trunk.norm
        self.width = trunk.embed_dim
        self.n_prefix = trunk.num_prefix_tokens
        self.embed_modules = [trunk.patch_embed, trunk.norm_pre]
        self.embed_params = [p for p in (getattr(trunk, "cls_token", None), getattr(trunk, "pos_embed", None))
                             if p is not None]

    def embed(self, x):
        tr = self.trunk
        x = tr._pos_embed(tr.patch_embed(x))
        x = tr.patch_drop(x)
        return tr.norm_pre(x)


def _norm_from(preprocess):
    for t in preprocess.transforms:
        if isinstance(t, T.Normalize):
            return tuple(t.mean), tuple(t.std)
    return (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)


def load_vit(backbone_name):
    """Image tower only, wrapped in an adapter, plus its normalisation."""
    lib, hub_id = BACKBONES[backbone_name]
    if lib == "open_clip":
        model, preprocess = open_clip.create_model_from_pretrained(hub_id)
        mean, std = _norm_from(preprocess)
        v = model.visual
        adapter = TimmAdapter(v.trunk) if hasattr(v, "trunk") else OpenCLIPAdapter(v)
    else:
        from transformers import CLIPImageProcessor, CLIPVisionModel

        vm = CLIPVisionModel.from_pretrained(hub_id, torch_dtype=torch.float32).vision_model
        ip = CLIPImageProcessor.from_pretrained(hub_id)
        mean, std = tuple(ip.image_mean), tuple(ip.image_std)
        adapter = HFCLIPAdapter(vm)
    return adapter, mean, std


class ViTMoEBackbone(nn.Module):
    """
    Drop-in replacement for zoommoe_model.MoEBackbone:
      forward(x, cond) -> (feature map B x C x 7 x 7, list of routing stats)
      freeze_until(stage), out_channels
    "layer3" MoE sits after block round(2L/3), "layer4" MoE after the last block,
    mirroring the end of ResNet stages 3 and 4. freeze_until("layer2") freezes
    the patch embedding and the first half of the blocks (ResNet: stem..layer2).
    """

    def __init__(self, cond_dim, cfg, adapter):
        super().__init__()
        self.vit = adapter
        L = len(adapter.blocks)
        self.grid = cfg.img_size // 32
        where = {"layer3": round(2 * L / 3) - 1, "layer4": L - 1}
        self.moe = nn.ModuleDict({
            str(where[s]): TokenMoE(adapter.width, cond_dim, cfg.n_experts, cfg.top_k, cfg.expert_ratio)
            for s in cfg.moe_stages if s in where
        })
        self.out_channels = adapter.width

    def vit_parameters(self):
        return list(self.vit.parameters())

    def freeze_until(self, stage):
        for p in self.vit.parameters():
            p.requires_grad = True
        if stage is None:
            return
        frozen = list(self.vit.embed_modules) + list(self.vit.blocks[: len(self.vit.blocks) // 2])
        for m in frozen:
            for p in m.parameters():
                p.requires_grad = False
        for p in getattr(self.vit, "embed_params", []):
            p.requires_grad = False

    def forward(self, x, cond):
        stats = []
        t = self.vit.embed(x)
        for i, blk in enumerate(self.vit.blocks):
            t = self.vit.run_block(blk, t)
            if str(i) in self.moe:
                t, s = self.moe[str(i)](t, cond)
                stats.append(s)
        t = self.vit.final_norm(t)
        p = t[:, self.vit.n_prefix:]                        # patch tokens
        B, N, C = p.shape
        g = int(round(N ** 0.5))
        feat = p.transpose(1, 2).reshape(B, C, g, g)
        if g != self.grid:                                  # ViT-B/16: 14x14 -> 7x7
            feat = F.adaptive_avg_pool2d(feat, self.grid)
        return feat, stats


BUILD_LOCK = threading.Lock()


def build_model(task, backbone_name):
    """zoommoe_model.ZoomMoE with its MoEBackbone swapped for the VLM backbone."""
    zm = TASK[task]["zm"]
    fields = {f.name for f in dataclasses.fields(zm.ZoomMoEConfig)}
    kw = dict(img_size=IMG_SIZE, n_experts=8, top_k=2, pretrained=False)
    if "n_classes" in fields:
        kw["n_classes"] = TASK[task]["n_classes"]
    cfg = zm.ZoomMoEConfig(**kw)
    with BUILD_LOCK:   # the swap is a module global; never build two models at once
        adapter, mean, std = load_vit(backbone_name)
        original = zm.MoEBackbone
        zm.MoEBackbone = lambda cond_dim, c: ViTMoEBackbone(cond_dim, c, adapter)
        try:
            model = zm.ZoomMoE(cfg)
        finally:
            zm.MoEBackbone = original
    return model, cfg, mean, std


# %% [Cell 6] Smoke test: every backbone x task, one forward/backward on random data
def smoke_test(task, backbone_name, dev):
    zm = TASK[task]["zm"]
    model, cfg, _, _ = build_model(task, backbone_name)
    model = model.to(dev)
    B, M = 4, 4
    images = torch.randn(B, M, 3, IMG_SIZE, IMG_SIZE, device=dev)
    mask = torch.ones(B, M, dtype=torch.bool, device=dev)
    mask[0, 3] = False
    n_cls = TASK[task]["n_classes"]
    label = (torch.randint(0, 2, (B,), device=dev).float() if n_cls == 1
             else torch.randint(0, n_cls, (B,), device=dev))
    batch = {"images": images, "mask": mask, "label": label, "patient_idx": torch.tensor([0, 0, 1, 2], device=dev)}
    with torch.autocast("cuda", dtype=torch.float16):
        out = model(images, mask)
        total, parts = zm.compute_losses(out, batch, cfg)
    total.backward()
    assert torch.isfinite(total)
    n_vit = sum(p.numel() for p in model.backbone.vit_parameters())
    n_all = sum(p.numel() for p in model.parameters())
    print(f"  {backbone_name:14s} {task:10s} logit {tuple(out['logit'].shape)}  "
          f"loss={float(total):.3f}  params: ViT {n_vit / 1e6:.1f}M, total {n_all / 1e6:.1f}M")
    del model, out, total
    gc.collect()
    torch.cuda.empty_cache()


print("smoke test:")
for _bb in BACKBONES:
    for _t in TASKS:
        smoke_test(_t, _bb, DEVICES[0])
print("smoke test passed")


# %% [Cell 7] Training (notebook-02 run_stage, per-GPU, with a ViT lr multiplier)
LOCK_PRINT = threading.Lock()
LAST_PROGRESS = [time.time()]   # updated on every log line; read by the stall watchdog


def log(*a):
    with LOCK_PRINT:
        print(*a, flush=True)
        LAST_PROGRESS[0] = time.time()


def run_stage(model, loader, cfg, zm, dev, epochs, lr, supervised, unsupervised, freeze, tag, log_every=50):
    model.backbone.freeze_until(freeze)
    vit_ids = {id(p) for p in model.backbone.vit_parameters()}
    vit_p = [p for p in model.parameters() if p.requires_grad and id(p) in vit_ids]
    rest_p = [p for p in model.parameters() if p.requires_grad and id(p) not in vit_ids]
    groups = [{"params": rest_p, "lr": lr}]
    max_lr = [lr]
    if vit_p:
        groups.append({"params": vit_p, "lr": lr * VIT_LR_MULT})
        max_lr.append(lr * VIT_LR_MULT)
    params = vit_p + rest_p
    opt = torch.optim.AdamW(groups, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=max_lr,
                                                total_steps=max(epochs * len(loader), 1), pct_start=0.3)
    scaler = torch.amp.GradScaler("cuda")

    for ep in range(epochs):
        model.train()
        running, t0 = [], time.time()
        for it, batch in enumerate(loader):
            batch = {k: v.to(dev, non_blocking=True) for k, v in batch.items()}
            frac = (ep * len(loader) + it) / max(epochs * len(loader), 1)
            lam = float(2.0 / (1.0 + np.exp(-10 * frac)) - 1.0)          # GRL ramp, as notebook 02
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
            if it % log_every == 0 and ep == 0:
                log(f"  [{tag}] ep{ep} it{it}: total={float(total):.3f}")
        mean = {k: float(np.mean([r[k] for r in running])) for k in running[0]}
        log(f"  [{tag}] epoch {ep + 1}/{epochs} ({time.time() - t0:.0f}s): "
            + "  ".join(f"{k}={v:.3f}" for k, v in mean.items()))


def evaluate_loader(model, loader, task, dev):
    res = TASK[task]["zm"].collect(model, loader, dev)
    return TASK[task]["bc"].evaluate(res["label"], res["prob"], res["patient_idx"]), res


def job_file(bb, task, fi):
    return JOBS_DIR / f"{safe_name(bb)}__{task}__fold{fi}.npz"


def train_job(bb, task, fi, dev):
    torch.cuda.set_device(dev)
    tag = f"{bb}|{task}|f{fi}|{dev}"
    s, fold, zm = TASK[task]["samples"], TASK[task]["folds"][fi], TASK[task]["zm"]
    binary = TASK[task]["n_classes"] == 1
    set_seed(SEED + fi)
    model, cfg, mean, std = build_model(task, bb)
    model = model.to(dev)
    n_params = sum(p.numel() for p in model.parameters())

    def loader(idx, train):
        ds = MultiScaleRAM(s.iloc[idx], train, binary,
                           scale_dropout=SCALE_DROPOUT if train else 0.0, seed=SEED + fi)
        if train:
            return GPUBatches(ds, BATCH, True, True, dev, mean, std, SEED + fi)
        return GPUBatches(ds, 32, False, False, dev, mean, std, SEED + fi)

    tr, va, te = loader(fold["train"], True), loader(fold["val"], False), loader(fold["test"], False)
    log(f"[{tag}] start: {({k: len(v) for k, v in fold.items()})} samples, {n_params / 1e6:.1f}M params")
    t0 = time.time()
    # the same three stages as notebook 02
    run_stage(model, tr, cfg, zm, dev, EPOCHS_S0, LR_S01, False, True, "layer2", f"{tag}-s0")
    run_stage(model, tr, cfg, zm, dev, EPOCHS_S1, LR_S01, True, True, "layer2", f"{tag}-s1")
    vm, _ = evaluate_loader(model, va, task, dev)
    log(f"[{tag}] after stage 1, val: " + str({k: round(v, 4) for k, v in vm.items()}))
    run_stage(model, tr, cfg, zm, dev, EPOCHS_S2, LR_S2, True, True, None, f"{tag}-s2")

    vm, _ = evaluate_loader(model, va, task, dev)
    tm, test_res = evaluate_loader(model, te, task, dev)
    minutes = (time.time() - t0) / 60
    log(f"[{tag}] val : " + str({k: round(v, 4) for k, v in vm.items()}))
    log(f"[{tag}] TEST: " + str({k: round(v, 4) for k, v in tm.items()}) + f"  ({minutes:.1f} min)")
    if SAVE_MODELS:
        # load with: model, cfg, _, _ = build_model(task, bb); model.load_state_dict(
        #               {k: v.float() for k, v in ckpt["state_dict"].items()})
        # (cfg is stored as a plain dict so the file loads without the zm module)
        torch.save({
            "state_dict": {k: v.half() if v.is_floating_point() else v for k, v in model.state_dict().items()},
            "cfg": dataclasses.asdict(cfg), "backbone": bb, "hub_id": BACKBONES[bb][1], "task": task,
            "fold": fi, "mean": mean, "std": std, "img_size": IMG_SIZE, "test_metrics": tm,
        }, MODELS_DIR / f"zoommoe_{safe_name(bb)}__{task}__fold{fi}.pt")
    # written last: its existence marks the job (and its checkpoint) as finished
    np.savez_compressed(job_file(bb, task, fi), **test_res, metrics=np.array([tm]), minutes=minutes,
                        n_params=n_params)
    del model, tr, va, te
    gc.collect()
    torch.cuda.empty_cache()


# %% [Cell 8] Run every (backbone, task, fold) on both GPUs
jobs = queue.Queue()
for bb in BACKBONES:
    for task in TASKS:
        for fi in range(len(TASK[task]["folds"])):
            if not job_file(bb, task, fi).exists():
                jobs.put((bb, task, fi))
print(f"{jobs.qsize()} jobs on {len(DEVICES)} GPU(s)")
FAILED = []


def gpu_worker(dev):
    while True:
        try:
            bb, task, fi = jobs.get_nowait()
        except queue.Empty:
            return
        try:
            train_job(bb, task, fi, dev)
        except Exception as e:
            FAILED.append((bb, task, fi, repr(e)))
            log(f"!! FAILED {bb} | {task} | fold {fi}: {e}\n{traceback.format_exc()}")
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
print(f"\nfinished in {(time.time() - T_START) / 3600:.2f} h; failed: {FAILED or 'none'}")


# %% [Cell 9] Results: ZoomMoE+VLM vs ZoomMoE+ResNet-50 (+ fine-tuned VLMs if provided)
rows, usage = [], {}
for bb in BACKBONES:
    for task in TASKS:
        n_f = len(TASK[task]["folds"])
        files = [job_file(bb, task, fi) for fi in range(n_f)]
        if not all(f.exists() for f in files):
            print(f"incomplete: {bb} | {task} ({sum(f.exists() for f in files)}/{n_f} folds)")
            continue
        for fi, f in enumerate(files):
            z = np.load(f, allow_pickle=True)
            rows.append({"task": task, "model": f"ZoomMoE + {bb}", "setting": "ZoomMoE (ours)",
                         "level": "sample (mag-avg)", "fold": fi, **z["metrics"][0],
                         "minutes": float(z["minutes"]), "params_M": float(z["n_params"]) / 1e6})
            usage.setdefault((bb, task), []).append(z["expert_usage"])

res = pd.DataFrame(rows)
res.to_csv(OUT / "zoommoe_vlm_fold_results.csv", index=False)

full = res.copy()
for task, path in ZOOMMOE_RESULTS.items():
    if path is not None and Path(path).exists():
        z = pd.read_csv(path)
        z = z[[c for c in ["acc", "prr", "f1_macro", "auc"] if c in z.columns]].copy()
        z["task"], z["model"], z["setting"], z["level"] = task, "ZoomMoE + ResNet-50", "ZoomMoE (ours)", "sample (mag-avg)"
        z["fold"] = range(len(z))
        full = pd.concat([full, z], ignore_index=True)
if PREV_RESULTS is not None and Path(PREV_RESULTS).exists():
    prev = pd.read_csv(PREV_RESULTS)
    prev = prev[prev.model != "ZoomMoE (ours)"]          # replaced by the two rows above
    full = pd.concat([prev, full], ignore_index=True)
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

if len(res):
    print("\ntraining cost (mean per fold):")
    display(res.groupby(["task", "model"])[["minutes", "params_M"]].mean().round(1))

# expert-usage heatmaps (magnification x expert), averaged over test sets and folds
if usage:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(usage), figsize=(4.5 * len(usage), 3.2), squeeze=False)
    for ax, ((bb, task), u) in zip(axes[0], usage.items()):
        ax.imshow(np.mean(u, 0), aspect="auto", cmap="viridis")
        ax.set_yticks(range(len(MAGS)), [f"{m}x" for m in MAGS])
        ax.set_xlabel("expert")
        ax.set_title(f"{bb} | {task}", fontsize=9)
    plt.tight_layout()
    plt.savefig(OUT / "expert_usage_vlm.png", dpi=200)
    plt.show()
print("outputs in", OUT, ":", sorted(p.name for p in OUT.iterdir()))
