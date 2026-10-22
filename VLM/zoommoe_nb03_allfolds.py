# =============================================================================
# 08 - Notebook-03 falsification tests on ALL 5 folds (binary + 8-class)
# =============================================================================
# Notebook 03 ran its tests on fold 0 only. This runs the same tests, with the
# same code, on every fold of both tasks, using the trained notebook-02
# checkpoints (datasets notebook2-articafts-breakhis / -mc):
#   1. magnification-order permutation   (AUC ordered - mean permuted)
#   2. patient-identity linear probe     (fused state h, z_inv)
#   3. expert specialisation             (I(E;M) bits + bootstrap CI, H(E))
#   4. zoom gain                         (metrics after 1..4 magnifications, AUC lift)
#   5. ordinal zoom structure            (z_spec distance 40x->100x/200x/400x increasing)
#   6. random-label control              (notebook 03 test 5, never run before; 3 epochs)
# Verification first: each checkpoint must reproduce notebook 02's TEST metrics
# for its fold, and each fold must match notebook 02's printed audit.
#
# Inference only (plus the short random-label control); one GPU; no threads.
# Kaggle: GPU T4, Internet ON (ResNet-50 weights for the random-label control).
# =============================================================================


# %% [Cell 1] Imports and configuration
import importlib.util
import os
import sys
import time
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")
T0 = time.time()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
RUN_RANDOM_LABEL = True
OUTDIR = Path("/kaggle/working/nb03_allfolds")
OUTDIR.mkdir(parents=True, exist_ok=True)

INPUTS = {
    "binary": dict(
        code=Path("/kaggle/input/datasets/nabinkoirala452/code-files"),
        art=Path("/kaggle/input/datasets/nabinkoirala452/articafts/artifacts"),
        ckpt=Path("/kaggle/input/datasets/nabinkoirala452/notebook2-articafts-breakhis/artifacts"),
        ckpt_name="zoommoe_fold{}.pt",
        results="zoommoe_results.csv",
    ),
    "multiclass": dict(
        code=Path("/kaggle/input/datasets/nabinkoirala452/code-files-mc"),
        art=Path("/kaggle/input/datasets/nabinkoirala452/articafts-notebook1-mc/artifacts"),
        ckpt=Path("/kaggle/input/datasets/nabinkoirala452/notebook2-articafts-breakhis-mc/artifacts"),
        ckpt_name="zoommoe_multiclass_fold{}.pt",
        results="zoommoe_multiclass_results.csv",
    ),
}

# printed by notebook 02 (cell 9): (train, val, test) sizes and patients per fold
NB02_FOLDS = {
    "binary": [((1638, 308, 461), (56, 10, 16)), ((1707, 309, 391), (56, 10, 16)), ((1622, 283, 502), (56, 10, 16)),
               ((1652, 233, 522), (55, 10, 17)), ((1608, 268, 531), (55, 10, 17))],
    "multiclass": [((1559, 324, 524), (56, 10, 16)), ((1616, 287, 504), (56, 10, 16)), ((1703, 228, 476), (55, 10, 17)),
                   ((1625, 302, 480), (56, 10, 16)), ((1701, 283, 423), (55, 10, 17))],
}
print("device:", DEVICE, "| torch", torch.__version__)


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


# %% [Cell 2] Load code, samples, folds per task (with checks)
AMP_OLD = "gates = torch.zeros_like(logits).scatter(-1, topi, topv.softmax(dim=-1))"
AMP_NEW = "gates = torch.zeros_like(logits).scatter(-1, topi, topv.softmax(dim=-1).to(logits.dtype))"
TASK = {}
for task, inp in INPUTS.items():
    code = resolve(inp["code"], "zoommoe_model.py")
    art = resolve(inp["art"], "samples_pool.pkl")
    ckpt_dir = resolve(inp["ckpt"], inp["ckpt_name"].format(0))
    bc = load_module(code / "breakhis_common.py", f"bc_{task}")
    src = (code / "zoommoe_model.py").read_text()
    if task == "binary":
        assert src.count(AMP_OLD) == 1
        src = src.replace(AMP_OLD, AMP_NEW)       # = the model binary notebook 02 trained (verified diff)
    else:
        assert src.count(AMP_NEW) == 1
    mpath = OUTDIR / f"zoommoe_model_{task}.py"
    mpath.write_text(src)
    zm = load_module(mpath, f"zm_{task}")
    samples = pd.read_pickle(art / "samples_pool.pkl")
    folds = list(np.load(art / "folds_all.npy", allow_pickle=True))
    for fi, (f, (n, p)) in enumerate(zip(folds, NB02_FOLDS[task])):
        got_n = tuple(len(f[k]) for k in ("train", "val", "test"))
        got_p = tuple(samples.iloc[f[k]].patient_id.nunique() for k in ("train", "val", "test"))
        assert (got_n, got_p) == (n, p), f"{task} fold {fi} differs from notebook 02: {got_n} {got_p}"
    ref = pd.read_csv(ckpt_dir / inp["results"]).set_index("fold")
    TASK[task] = dict(bc=bc, zm=zm, samples=samples, folds=folds, ckpt_dir=ckpt_dir, ref=ref,
                      ckpt_name=inp["ckpt_name"])
    print(f"[{task}] code={code} | art={art} | ckpt={ckpt_dir} | folds match notebook 02")


def load_ckpt(task, fi):
    t = TASK[task]
    sys.modules["zoommoe_model"] = t["zm"]        # the pickled cfg references module "zoommoe_model"
    ck = torch.load(t["ckpt_dir"] / t["ckpt_name"].format(fi), map_location=DEVICE, weights_only=False)
    model = t["zm"].ZoomMoE(ck["cfg"]).to(DEVICE)
    model.load_state_dict(ck["state_dict"])
    return model.eval(), ck["cfg"]


# %% [Cell 3] The notebook-03 tests (same code, functions take task/model explicitly)
def permutation_test(t, model, loader, res, n_perm=6, seed=SEED):
    bc, zm = t["bc"], t["zm"]
    rng = np.random.default_rng(seed)
    rows = [{"order": "0,1,2,3", "kind": "ordered", **bc.evaluate(res["label"], res["prob"], res["patient_idx"])}]
    seen = {(0, 1, 2, 3)}
    for _ in range(n_perm):
        order = tuple(rng.permutation(4).tolist())
        if order in seen:
            continue
        seen.add(order)
        r = zm.collect(model, loader, DEVICE, order=list(order))
        rows.append({"order": ",".join(map(str, order)), "kind": "permuted",
                     **bc.evaluate(r["label"], r["prob"], r["patient_idx"])})
    r = zm.collect(model, loader, DEVICE, order=[3, 2, 1, 0])
    rows.append({"order": "3,2,1,0", "kind": "reversed", **bc.evaluate(r["label"], r["prob"], r["patient_idx"])})
    df = pd.DataFrame(rows)
    ordered = float(df.loc[df.kind == "ordered", "auc"].iloc[0])
    permuted = float(df.loc[df.kind == "permuted", "auc"].mean())
    reversed_ = float(df.loc[df.kind == "reversed", "auc"].iloc[0])
    return df, ordered - permuted, ordered - reversed_


def patient_probe(features, patient_idx, seed=SEED, min_per_patient=6):
    rng = np.random.default_rng(seed)
    tr, te = [], []
    for p in np.unique(patient_idx):
        loc = np.flatnonzero(patient_idx == p)
        if len(loc) < min_per_patient:
            continue
        rng.shuffle(loc)
        cut = len(loc) // 2
        tr.extend(loc[:cut])
        te.extend(loc[cut:])
    if len(set(patient_idx[tr])) < 2:
        return {"probe_acc": np.nan, "chance": np.nan, "ratio": np.nan}
    scaler = StandardScaler().fit(features[tr])
    clf = LogisticRegression(max_iter=2000).fit(scaler.transform(features[tr]), patient_idx[tr])
    acc = float(clf.score(scaler.transform(features[te]), patient_idx[te]))
    chance = 1.0 / len(np.unique(patient_idx[tr]))
    return {"probe_acc": acc, "chance": chance, "ratio": acc / chance}


def mutual_information_bits(table):
    q = table / table.sum(axis=1, keepdims=True).clip(1e-9)
    marginal = q.mean(axis=0)
    marginal = marginal / marginal.sum()
    ent = lambda p: float(-(p * np.log2(np.clip(p, 1e-12, None))).sum())
    return ent(marginal) - float(np.mean([ent(row) for row in q])), ent(marginal)


@torch.no_grad()
def per_batch_usage(t, model, loader, max_batches=40):
    tables = []
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        out = model(batch["images"].to(DEVICE), batch["mask"].to(DEVICE))
        tables.append(t["zm"].expert_usage_table(out["route"], model.cfg.n_experts).numpy())
    return np.stack(tables)


def zoom_gain(t, res):
    rows = []
    for k in range(res["step_prob"].shape[1]):
        m = t["bc"].evaluate(res["label"], res["step_prob"][:, k], res["patient_idx"])
        m["mags_seen"] = k + 1
        m["mean_utility"] = float(res["utility"][:, k].mean())
        rows.append(m)
    return pd.DataFrame(rows)


def ordinal_structure(res):
    z_spec, valid = res["z_spec"], res["mask"]
    d = np.stack([np.linalg.norm(z_spec[:, m] - z_spec[:, 0], axis=-1) for m in range(1, 4)], axis=1)
    ok = valid[:, 0][:, None] & valid[:, 1:]
    means = [float(d[:, k][ok[:, k]].mean()) for k in range(3)]
    return means, all(means[k] < means[k + 1] for k in range(2))


def random_label_control(t, cfg, fi, test_loader):
    """notebook 03 test 5, verbatim (3 epochs, labels shuffled inside the training fold)"""
    bc, zm, samples, folds = t["bc"], t["zm"], t["samples"], t["folds"]
    bc.set_seed(SEED + fi)
    shuffled = samples.copy()
    rng = np.random.default_rng(SEED)
    tr_idx = folds[fi]["train"]
    perm = rng.permutation(tr_idx)
    shuffled.loc[shuffled.index[tr_idx], "label"] = samples["label"].values[perm]
    ctrl_cfg = replace(cfg, pretrained=True)
    ctrl = zm.ZoomMoE(ctrl_cfg).to(DEVICE)
    tr_loader = DataLoader(bc.make_image_dataset(shuffled.iloc[tr_idx], train=True, img_size=cfg.img_size),
                           batch_size=8, shuffle=True, num_workers=2, drop_last=True)
    opt = torch.optim.AdamW([p for p in ctrl.parameters() if p.requires_grad], lr=3e-4)
    ctrl.train()
    for _ in range(3):
        for batch in tr_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            out = ctrl(batch["images"], batch["mask"])
            total, _ = zm.compute_losses(out, batch, ctrl_cfg)
            opt.zero_grad(set_to_none=True)
            total.backward()
            opt.step()
    r = zm.collect(ctrl, test_loader, DEVICE)
    m = bc.evaluate(r["label"], r["prob"], r["patient_idx"])
    del ctrl
    torch.cuda.empty_cache()
    return m


# %% [Cell 4] Run every fold of both tasks
rows, zoom_rows, perm_rows, usage_tables = [], [], [], {}
for task in ("binary", "multiclass"):
    t = TASK[task]
    for fi in range(5):
        tf = time.time()
        model, cfg = load_ckpt(task, fi)
        test_idx = t["folds"][fi]["test"]
        loader = DataLoader(t["bc"].make_image_dataset(t["samples"].iloc[test_idx], train=False, img_size=cfg.img_size),
                            batch_size=8, shuffle=False, num_workers=2)
        res = t["zm"].collect(model, loader, DEVICE)
        base = t["bc"].evaluate(res["label"], res["prob"], res["patient_idx"])
        ref = t["ref"].loc[fi]
        diff = max(abs(base[k] - ref[k]) for k in ("acc", "f1_macro", "prr", "auc"))
        assert diff < 0.01, f"{task} fold {fi}: checkpoint does not reproduce notebook 02 ({base} vs {ref.to_dict()})"

        perm_df, gap, gap_rev = permutation_test(t, model, loader, res)
        perm_df["task"], perm_df["fold"] = task, fi
        perm_rows.append(perm_df)
        probe_h = patient_probe(res["state"], res["patient_idx"])
        probe_z = patient_probe(res["z_inv"].reshape(len(res["z_inv"]), -1), res["patient_idx"])
        mi, h_marg = mutual_information_bits(res["expert_usage"])
        bt = per_batch_usage(t, model, loader)
        boot = [mutual_information_bits(bt[np.random.default_rng(s).integers(0, len(bt), len(bt))].mean(0))[0]
                for s in range(200)]
        zc = zoom_gain(t, res)
        zc["task"], zc["fold"] = task, fi
        zoom_rows.append(zc)
        lift = float(zc.loc[zc.mags_seen == 4, "auc"].iloc[0] - zc.loc[zc.mags_seen == 1, "auc"].iloc[0])
        means, monotone = ordinal_structure(res)
        usage_tables[f"{task}_fold{fi}"] = res["expert_usage"]
        row = {"task": task, "fold": fi, **{f"test_{k}": v for k, v in base.items()},
               "max_diff_vs_nb02": diff, "order_gap_auc": gap, "reversed_gap_auc": gap_rev,
               "probe_h_ratio": probe_h["ratio"], "probe_h_acc": probe_h["probe_acc"],
               "probe_zinv_ratio": probe_z["ratio"], "mi_bits": mi,
               "mi_ci_low": float(np.percentile(boot, 2.5)), "mi_ci_high": float(np.percentile(boot, 97.5)),
               "H_marginal_bits": h_marg, "zoom_auc_lift": lift,
               "ordinal_d100": means[0], "ordinal_d200": means[1], "ordinal_d400": means[2],
               "ordinal_monotone": monotone}
        if RUN_RANDOM_LABEL:
            rl = random_label_control(t, cfg, fi, loader)
            row.update({f"randlabel_{k}": v for k, v in rl.items()})
        rows.append(row)
        pd.DataFrame(rows).to_csv(OUTDIR / "nb03_allfolds_per_fold.csv", index=False)
        print(f"[{task} f{fi}] reproduces nb02 (max diff {diff:.4f}) | order gap {gap:+.4f} | "
              f"probe {probe_h['ratio']:.1f}x | I(E;M) {mi:.3f} | zoom lift {lift:+.4f} | ordinal {monotone}"
              + (f" | random-label AUC {row['randlabel_auc']:.3f}" if RUN_RANDOM_LABEL else "")
              + f"  ({(time.time() - tf) / 60:.1f} min, total {(time.time() - T0) / 60:.0f} min)", flush=True)
        del model
        torch.cuda.empty_cache()

pd.concat(perm_rows).to_csv(OUTDIR / "test1_permutation_all.csv", index=False)
pd.concat(zoom_rows).to_csv(OUTDIR / "test4_zoom_gain_all.csv", index=False)
np.savez(OUTDIR / "expert_usage_all.npz", **usage_tables)


# %% [Cell 5] Summary across folds (mean ± std, folds passing notebook 03's criteria)
df = pd.DataFrame(rows)
crit = {
    "order_gap_auc": ("order permutation: AUC gap >= 0.01", lambda x: x >= 0.01),
    "probe_h_ratio": ("patient probe on h: < 5x chance", lambda x: x < 5),
    "mi_bits": ("expert specialisation: I(E;M) >= 0.05 bits", lambda x: x >= 0.05),
    "zoom_auc_lift": ("zoom gain: AUC lift >= 0.01", lambda x: x >= 0.01),
    "ordinal_monotone": ("ordinal structure monotone", lambda x: bool(x)),
}
if RUN_RANDOM_LABEL:
    crit["randlabel_auc"] = ("random-label control: AUC <= 0.60", lambda x: x <= 0.60)
summ = []
for task, g in df.groupby("task", sort=False):
    for col, (name, ok) in crit.items():
        vals = g[col]
        stat = f"{vals.mean():.3f} ± {vals.std():.3f}" if col != "ordinal_monotone" else f"{int(vals.sum())}/5 folds"
        summ.append({"task": task, "test": name, "value (mean ± std over folds)": stat,
                     "folds passing": f"{int(vals.map(ok).sum())}/5"})
summ = pd.DataFrame(summ)
display(summ)
summ.to_csv(OUTDIR / "nb03_allfolds_summary.csv", index=False)

zz = pd.concat(zoom_rows).groupby(["task", "mags_seen"])[["acc", "f1_macro", "auc"]].mean().round(4)
print("\nzoom-gain curve, mean over folds:")
display(zz)
zz.to_csv(OUTDIR / "test4_zoom_gain_mean.csv")
print(f"\ndone in {(time.time() - T0) / 60:.0f} min; outputs: {sorted(p.name for p in OUTDIR.iterdir())}")
