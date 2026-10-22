"""ZoomMoE: zoom-conditioned MoE backbone + progressive zoom refinement."""

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

N_MAGS = 4


@dataclass
class ZoomMoEConfig:
    img_size: int = 224
    n_mags: int = N_MAGS
    mag_dim: int = 32
    state_dim: int = 256
    tok_dim: int = 256
    cond_state_dim: int = 64
    n_queries: int = 8
    attn_heads: int = 4
    proj_dim: int = 128
    n_experts: int = 8
    top_k: int = 2
    expert_ratio: int = 4
    moe_stages: Tuple[str, ...] = ("layer3", "layer4")
    pretrained: bool = True
    dropout: float = 0.3
    # loss weights
    w_align: float = 1.0
    w_ordinal: float = 0.5
    w_disent: float = 0.5
    w_advmag: float = 0.2
    w_supcon: float = 0.5
    w_deep: float = 0.3
    w_bal: float = 0.01
    w_spec: float = 0.05
    w_zloss: float = 1e-3
    tau: float = 0.1
    ordinal_margin: float = 0.05


# ---------------------------------------------------------------- gradient reversal
class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.lambd * grad, None


def grad_reverse(x, lambd=1.0):
    return _GradReverse.apply(x, lambd)


# ---------------------------------------------------------------- Conv-MoE
class ConvMoE(nn.Module):
    """
    Sparse top-k mixture of convolutional experts with per-location routing.

    The router input is [local feature ; conditioning], where conditioning is
    [magnification embedding ; projected zoom state h_{k-1}]. Experts are
    inverted bottlenecks so that running all of them densely stays affordable;
    a production version would gather tokens per expert instead.
    """

    def __init__(self, channels: int, cond_dim: int, n_experts: int = 8, top_k: int = 2, ratio: int = 4):
        super().__init__()
        self.n_experts, self.top_k = n_experts, top_k
        hidden = max(channels // ratio, 32)
        self.norm = nn.GroupNorm(1, channels)
        self.router = nn.Linear(channels + cond_dim, n_experts)
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(channels, hidden, 1, bias=False),
                    nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False),
                    nn.GELU(),
                    nn.Conv2d(hidden, channels, 1, bias=False),
                )
                for _ in range(n_experts)
            ]
        )
        nn.init.zeros_(self.router.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        B, C, H, W = x.shape
        N = H * W
        u = self.norm(x).permute(0, 2, 3, 1).reshape(B, N, C)
        c = cond.unsqueeze(1).expand(-1, N, -1)
        logits = self.router(torch.cat([u, c], dim=-1))

        if self.training:
            logits = logits + torch.randn_like(logits) / self.n_experts

        topv, topi = logits.topk(self.top_k, dim=-1)
        # FIX: Cast softmax back to the original dtype to avoid AMP scatter mismatch
        gates = torch.zeros_like(logits).scatter(-1, topi, topv.softmax(dim=-1).to(logits.dtype))

        out = x
        for e, expert in enumerate(self.experts):
            g = gates[..., e]
            if not self.training and float(g.detach().abs().sum()) == 0.0:
                continue
            out = out + g.reshape(B, 1, H, W) * expert(x)

        stats = {
            "gates": gates.mean(dim=(0, 1)),
            "frac": (gates > 0).float().mean(dim=(0, 1)),
            "zloss": logits.logsumexp(dim=-1).pow(2).mean(),
        }
        return out, stats


# ---------------------------------------------------------------- backbone
class MoEBackbone(nn.Module):
    """ResNet-50 with Conv-MoE blocks inserted at the end of the last stages."""

    def __init__(self, cond_dim: int, cfg: ZoomMoEConfig):
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if cfg.pretrained else None
        r = models.resnet50(weights=weights)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = r.layer1, r.layer2, r.layer3, r.layer4

        mk = lambda ch: ConvMoE(ch, cond_dim, cfg.n_experts, cfg.top_k, cfg.expert_ratio)
        self.moe3 = mk(1024) if "layer3" in cfg.moe_stages else None
        self.moe4 = mk(2048) if "layer4" in cfg.moe_stages else None
        self.out_channels = 2048

    def freeze_until(self, stage: Optional[str]):
        order = ["stem", "layer1", "layer2", "layer3", "layer4"]
        if stage is None:
            for p in self.parameters():
                p.requires_grad = True
            return
        stop = order.index(stage)
        for i, name in enumerate(order):
            for p in getattr(self, name).parameters():
                p.requires_grad = i > stop

    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        stats = []
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if self.moe3 is not None:
            x, s = self.moe3(x, cond)
            stats.append(s)
        x = self.layer4(x)
        if self.moe4 is not None:
            x, s = self.moe4(x, cond)
            stats.append(s)
        return x, stats


# ---------------------------------------------------------------- progressive zoom
class ProgressiveZoom(nn.Module):
    """h_k = h_{k-1} + g_k * delta_k, driven by cross-attention into F_{m_k}."""

    def __init__(self, cfg: ZoomMoEConfig):
        super().__init__()
        self.n_q, self.tok_dim = cfg.n_queries, cfg.tok_dim
        self.to_q = nn.Sequential(
            nn.Linear(cfg.state_dim, cfg.tok_dim),
            nn.GELU(),
            nn.Linear(cfg.tok_dim, cfg.n_queries * cfg.tok_dim),
        )
        self.attn = nn.MultiheadAttention(cfg.tok_dim, cfg.attn_heads, batch_first=True)
        self.ln_q = nn.LayerNorm(cfg.tok_dim)
        self.ln_t = nn.LayerNorm(cfg.tok_dim)
        joint = cfg.state_dim + cfg.tok_dim
        self.gate = nn.Sequential(nn.Linear(joint, cfg.state_dim), nn.Sigmoid())
        self.delta = nn.Sequential(
            nn.Linear(joint, cfg.state_dim), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(cfg.state_dim, cfg.state_dim),
        )
        self.step_head = nn.Linear(cfg.state_dim, 1)

    def step(self, h: torch.Tensor, tokens: torch.Tensor):
        B = h.shape[0]
        q = self.ln_q(self.to_q(h).view(B, self.n_q, self.tok_dim))
        t = self.ln_t(tokens)
        a, attn_w = self.attn(q, t, t, need_weights=True)
        a = a.mean(dim=1)
        hc = torch.cat([h, a], dim=-1)
        upd = self.gate(hc) * self.delta(hc)
        h_new = h + upd
        utility = upd.abs().sum(dim=-1)
        return h_new, self.step_head(h_new).squeeze(-1), utility, attn_w


# ---------------------------------------------------------------- full model
class ZoomMoE(nn.Module):
    def __init__(self, cfg: ZoomMoEConfig):
        super().__init__()
        self.cfg = cfg
        cond_dim = cfg.mag_dim + cfg.cond_state_dim

        self.mag_emb = nn.Embedding(cfg.n_mags, cfg.mag_dim)
        # cond_state_dim == 0 removes the zoom-state feedback edge (ablation)
        self.state_to_cond = (
            nn.Sequential(
                nn.LayerNorm(cfg.state_dim), nn.Linear(cfg.state_dim, cfg.cond_state_dim), nn.GELU()
            )
            if cfg.cond_state_dim > 0
            else None
        )
        self.backbone = MoEBackbone(cond_dim, cfg)
        C = self.backbone.out_channels

        self.tok_proj = nn.Conv2d(C, cfg.tok_dim, 1)
        n_tok = (cfg.img_size // 32) ** 2
        self.pos = nn.Parameter(torch.zeros(1, n_tok, cfg.tok_dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.zoom_tok = nn.Embedding(cfg.n_mags, cfg.tok_dim)

        self.pzr = ProgressiveZoom(cfg)
        self.h0 = nn.Parameter(torch.zeros(1, cfg.state_dim))
        self.head = nn.Sequential(nn.LayerNorm(cfg.state_dim), nn.Dropout(cfg.dropout), nn.Linear(cfg.state_dim, 1))

        mkproj = lambda: nn.Sequential(nn.Linear(C, 512), nn.GELU(), nn.Linear(512, cfg.proj_dim))
        self.proj_inv = mkproj()
        self.proj_spec = mkproj()
        self.mag_clf_inv = nn.Linear(cfg.proj_dim, cfg.n_mags)
        self.mag_clf_spec = nn.Linear(cfg.proj_dim, cfg.n_mags)

    def forward(
        self,
        images: torch.Tensor,          # B, M, 3, H, W
        mask: torch.Tensor,            # B, M  (bool)
        order: Optional[Sequence[int]] = None,
        grl_lambda: float = 1.0,
        need_attn: bool = False,
    ) -> Dict[str, torch.Tensor]:
        B, M = images.shape[0], images.shape[1]
        order = list(range(M)) if order is None else list(order)
        dev = images.device

        h = self.h0.expand(B, -1)
        per_mag: Dict[int, Dict[str, torch.Tensor]] = {}
        route: List[Tuple[int, List[Dict[str, torch.Tensor]]]] = []
        attn_maps: Dict[int, torch.Tensor] = {}

        for m in order:
            mag_id = torch.full((B,), m, dtype=torch.long, device=dev)
            cond_parts = [self.mag_emb(mag_id)]
            if self.state_to_cond is not None:
                cond_parts.append(self.state_to_cond(h))
            cond = torch.cat(cond_parts, dim=-1)

            feat, stats = self.backbone(images[:, m], cond)
            route.append((m, stats))

            pooled = feat.mean(dim=(2, 3))
            z_inv = F.normalize(self.proj_inv(pooled), dim=-1)
            z_spec = F.normalize(self.proj_spec(pooled), dim=-1)

            tokens = self.tok_proj(feat).flatten(2).transpose(1, 2)
            tokens = tokens + self.pos + self.zoom_tok(mag_id).unsqueeze(1)

            h_new, step_logit, utility, attn_w = self.pzr.step(h, tokens)
            keep = mask[:, m].float().unsqueeze(-1)
            h = keep * h_new + (1.0 - keep) * h

            per_mag[m] = {
                "z_inv": z_inv,
                "z_spec": z_spec,
                "step_logit": step_logit,
                "utility": utility * mask[:, m].float(),
            }
            if need_attn:
                attn_maps[m] = attn_w.detach()

        stack = lambda key: torch.stack([per_mag[m][key] for m in range(M)], dim=1)
        z_inv_all, z_spec_all = stack("z_inv"), stack("z_spec")

        out = {
            "logit": self.head(h).squeeze(-1),
            "state": h,
            "z_inv": z_inv_all,
            "z_spec": z_spec_all,
            "step_logits": stack("step_logit"),
            "utility": stack("utility"),
            "mag_logits_inv": self.mag_clf_inv(grad_reverse(z_inv_all, grl_lambda)),
            "mag_logits_spec": self.mag_clf_spec(z_spec_all),
            "route": route,
        }
        if need_attn:
            out["attn"] = attn_maps
        return out


# ---------------------------------------------------------------- losses
def info_nce_patient_safe(za, zb, patient_idx, tau=0.1):
    """Positives on the diagonal; same-patient off-diagonals removed from the
    negative set so the model cannot win by encoding slide/stain identity."""
    za, zb = F.normalize(za.float(), dim=-1), F.normalize(zb.float(), dim=-1)
    sim = za @ zb.t() / tau
    same = patient_idx[:, None] == patient_idx[None, :]
    eye = torch.eye(len(za), dtype=torch.bool, device=za.device)
    sim = sim.masked_fill(same & ~eye, float("-inf"))
    target = torch.arange(len(za), device=za.device)
    return F.cross_entropy(sim, target)


def crossmag_align_loss(z_inv, mask, patient_idx, tau=0.1, min_pairs=4):
    """Every ordered pair of magnifications of the same sample is a positive."""
    M = z_inv.shape[1]
    total, n = z_inv.new_zeros(()), 0
    for a in range(M):
        for b in range(a + 1, M):
            sel = mask[:, a] & mask[:, b]
            if int(sel.sum()) < min_pairs:
                continue
            za, zb, pid = z_inv[sel, a], z_inv[sel, b], patient_idx[sel]
            total = total + 0.5 * (
                info_nce_patient_safe(za, zb, pid, tau) + info_nce_patient_safe(zb, za, pid, tau)
            )
            n += 1
    return total / max(n, 1)


def zoom_ordinal_loss(z_spec, mask, margin=0.05):
    """d(z_40, z_100) < d(z_40, z_200) < d(z_40, z_400) on the scale-specific part."""
    anchor_ok = mask[:, 0]
    d = torch.norm(z_spec[:, 1:] - z_spec[:, :1], dim=-1)          # B, M-1
    valid = mask[:, 1:] & anchor_ok.unsqueeze(1)
    total, n = z_spec.new_zeros(()), 0
    for k in range(d.shape[1] - 1):
        v = valid[:, k] & valid[:, k + 1]
        if int(v.sum()) == 0:
            continue
        total = total + F.relu(d[v, k] - d[v, k + 1] + margin).mean()
        n += 1
    return total / max(n, 1)


def disentangle_loss(z_inv, z_spec, mask):
    """Squared cross-covariance between the invariant and specific subspaces."""
    a, b = z_inv[mask], z_spec[mask]
    if a.shape[0] < 4:
        return z_inv.new_zeros(())
    a = (a - a.mean(0)) / (a.std(0) + 1e-5)
    b = (b - b.mean(0)) / (b.std(0) + 1e-5)
    c = (a.t() @ b) / a.shape[0]
    return c.pow(2).sum() / c.numel()


def mag_adversary_losses(out, mask):
    """z_inv must NOT predict magnification (through GRL); z_spec must."""
    M = out["z_inv"].shape[1]
    tgt = torch.arange(M, device=mask.device).unsqueeze(0).expand(mask.shape[0], -1)
    sel = mask.reshape(-1)
    li = F.cross_entropy(out["mag_logits_inv"].reshape(-1, M)[sel], tgt.reshape(-1)[sel])
    ls = F.cross_entropy(out["mag_logits_spec"].reshape(-1, M)[sel], tgt.reshape(-1)[sel])
    return li, ls


def supcon_cross_patient(h, labels, patient_idx, tau=0.1):
    """Positives are same-class, DIFFERENT-patient. Same-patient pairs are
    excluded entirely: too easy as positives, misleading as negatives."""
    z = F.normalize(h.float(), dim=-1)
    sim = z @ z.t() / tau
    n = len(z)
    eye = torch.eye(n, dtype=torch.bool, device=z.device)
    same_pat = patient_idx[:, None] == patient_idx[None, :]
    valid = ~eye & ~same_pat
    pos = valid & (labels[:, None] == labels[None, :])

    logits = sim.masked_fill(~valid, torch.finfo(sim.dtype).min / 4)
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    cnt = pos.sum(1)
    keep = cnt > 0
    if int(keep.sum()) == 0:
        return h.new_zeros(())
    loss = -(log_prob * pos).sum(1)[keep] / cnt[keep]
    return loss.mean()


def routing_losses(route, n_experts, device=None):
    """
    L_bal   : Switch-style, minimised at uniform marginal expert usage.
    L_spec  : -I(E;M), so usage is informative about magnification.
    L_zloss : router logit magnitude, for numerical stability.
    """
    per_mag = defaultdict(list)
    bal, zl, nb = None, None, 0
    for m, blocks in route:
        for s in blocks:
            term = n_experts * (s["frac"] * s["gates"]).sum()
            bal = term if bal is None else bal + term
            zl = s["zloss"] if zl is None else zl + s["zloss"]
            per_mag[m].append(s["gates"])
            nb += 1
    if nb == 0:  # MoE disabled (ablation)
        zero = torch.zeros((), device=device)
        return zero, zero, zero

    q_m = torch.stack([torch.stack(v).mean(0) for v in per_mag.values()])
    q_m = q_m / q_m.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    q = q_m.mean(0)
    q = q / q.sum().clamp(min=1e-8)
    ent = lambda p: -(p * (p + 1e-9).log()).sum(-1)
    mi = ent(q) - ent(q_m).mean()
    return bal / nb, -mi, zl / nb


def expert_usage_table(route, n_experts, n_mags=N_MAGS):
    """n_mags x n_experts gate-mass table, for the interpretability heatmap."""
    tab = torch.zeros(n_mags, n_experts)
    cnt = torch.zeros(n_mags)
    for m, blocks in route:
        for s in blocks:
            tab[m] += s["gates"].detach().float().cpu()
            cnt[m] += 1
    return tab / cnt.clamp(min=1).unsqueeze(1)


# ---------------------------------------------------------------- objective
def compute_losses(out, batch, cfg: ZoomMoEConfig, supervised=True, unsupervised=True):
    mask = batch["mask"]
    y = batch["label"]
    pid = batch["patient_idx"]
    parts: Dict[str, torch.Tensor] = {}

    if supervised:
        parts["cls"] = F.binary_cross_entropy_with_logits(out["logit"], y)
        valid = mask.reshape(-1)
        deep_tgt = y.unsqueeze(1).expand_as(out["step_logits"]).reshape(-1)[valid]
        parts["deep"] = cfg.w_deep * F.binary_cross_entropy_with_logits(
            out["step_logits"].reshape(-1)[valid], deep_tgt
        )
        parts["supcon"] = cfg.w_supcon * supcon_cross_patient(out["state"], y, pid, cfg.tau)

    if unsupervised:
        parts["align"] = cfg.w_align * crossmag_align_loss(out["z_inv"], mask, pid, cfg.tau)
        parts["ordinal"] = cfg.w_ordinal * zoom_ordinal_loss(out["z_spec"], mask, cfg.ordinal_margin)
        parts["disent"] = cfg.w_disent * disentangle_loss(out["z_inv"], out["z_spec"], mask)
        li, ls = mag_adversary_losses(out, mask)
        parts["mag_adv"] = cfg.w_advmag * li
        parts["mag_spec"] = cfg.w_advmag * ls

    bal, spec, zl = routing_losses(out["route"], cfg.n_experts, out["logit"].device)
    parts["bal"] = cfg.w_bal * bal
    parts["moe_spec"] = cfg.w_spec * spec
    parts["zloss"] = cfg.w_zloss * zl

    total = sum(parts.values())
    return total, {k: float(v.detach()) for k, v in parts.items()}


# ---------------------------------------------------------------- inference helper
@torch.no_grad()
def collect(model, loader, device, order=None, max_batches=None):
    """Run the model over a loader and return everything notebook 03 needs."""
    model.eval()
    acc = defaultdict(list)
    usage, n_batch = None, 0
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        images = batch["images"].to(device)
        mask = batch["mask"].to(device)
        out = model(images, mask, order=order)
        acc["prob"].append(torch.sigmoid(out["logit"]).cpu().numpy())
        acc["step_prob"].append(torch.sigmoid(out["step_logits"]).cpu().numpy())
        acc["utility"].append(out["utility"].cpu().numpy())
        acc["state"].append(out["state"].cpu().numpy())
        acc["z_inv"].append(out["z_inv"].cpu().numpy())
        acc["z_spec"].append(out["z_spec"].cpu().numpy())
        acc["label"].append(batch["label"].numpy())
        acc["patient_idx"].append(batch["patient_idx"].numpy())
        acc["mask"].append(batch["mask"].numpy())
        tab = expert_usage_table(out["route"], model.cfg.n_experts)
        usage = tab if usage is None else usage + tab
        n_batch += 1
    result = {k: np.concatenate(v) for k, v in acc.items()}
    result["expert_usage"] = (usage / max(n_batch, 1)).numpy()
    return result