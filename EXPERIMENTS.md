# Experiments: runs, status, where things are

Guide's requests: (2026-10-19) "consolidate all results into an 8-page IEEE Transactions paper draft;
also do experiments with VLMs (small models) for classification for our comparison", then
(2026-10-21) "try fine-tuning and add proposed ZoomMoE to VLMs as you did for CNNs".

## Runs

| # | Experiment | Script | Kaggle notebook | Status |
|---|---|---|---|---|
| 01–03 | Protocol + baselines, ZoomMoE training, analysis | notebooks in [Binary Class/](Binary%20Class/), [MultiClass/](MultiClass/) | nabinkoirala452/mc-0x-…, breakhis-0x-… | done (Sept 2026) |
| 04 | Zero-shot + linear probe + generative VLMs | [vlm_breakhis_classification.py](VLM/vlm_breakhis_classification.py) | nabinkoirala452/vlm-breakhis-classification (local copy [VLM/vlm-breakhis-classification.ipynb](VLM/vlm-breakhis-classification.ipynb)) | done 2026-10-20, ~5 h |
| 05 | Full fine-tuning of 5 VLM image encoders | [vlm_finetune.py](VLM/vlm_finetune.py) | [nabinkoirala452/vlm-finetune](https://www.kaggle.com/code/nabinkoirala452/vlm-finetune) | done 2026-10-21, 2.41 h, 0 failures |
| 06 | ZoomMoE with a VLM backbone (QuiltNet, PLIP) | [zoommoe_vlm.py](VLM/zoommoe_vlm.py) | [nabinkoirala53/zoommoe-vlm](https://www.kaggle.com/code/nabinkoirala53/zoommoe-vlm) | done 2026-10-22, 6.69 h, 0 failures (QuiltNet + PLIP, both tasks, ~40 min/fold) |
| 07 | ZoomMoE training-time ablations, binary | [zoommoe_ablation.py](VLM/zoommoe_ablation.py), pushed via API from [VLM/kaggle_runs/zoommoe-ablation-binary/](VLM/kaggle_runs/zoommoe-ablation-binary/) | [nabinkoirala45/zoommoe-ablation-binary](https://www.kaggle.com/code/nabinkoirala45/zoommoe-ablation-binary) (Gmail nabin079bme023 / pulchowk) | done 2026-10-22, 9.14 h, 30/30 jobs (6 variants × 5 folds), 0 failures, ~35 min/job; all input checks passed |
| 08 | Notebook-03 tests on all 5 folds, both tasks (+ random-label control) | [zoommoe_nb03_allfolds.py](VLM/zoommoe_nb03_allfolds.py) | [nabinkoirala452/zoommoe-nb03-allfolds](https://www.kaggle.com/code/nabinkoirala452/zoommoe-nb03-allfolds) (Gmail nabinkoirala53) | done 2026-10-22, 98 min, 10/10 folds; every checkpoint reproduced notebook 02 exactly |
| 09 | Same-backbone baselines: ResNet-50 late_avg / mean_fusion / concat_fusion, ZoomMoE recipe, both tasks | [zoommoe_baselines.py](VLM/zoommoe_baselines.py) | [nabinkoirala53/zoommoe-baselines-resnet50](https://www.kaggle.com/code/nabinkoirala53/zoommoe-baselines-resnet50) (Gmail nabinkoirala452) | done 2026-10-22, 4.13 h, 30/30 jobs, 0 failures, ~16 min/job |

## Kaggle accounts

`.env` labels are Gmail names; the Kaggle usernames differ:

| Gmail (label) | Kaggle user | Used for |
|---|---|---|
| nabinkoirala53 | nabinkoirala452 | 04, 05, 08; owns all the datasets |
| nabinkoirala452 | nabinkoirala53 | 06, 09 |
| nabin079bme023 (pulchowk) | nabinkoirala45 | 07 (20.9 h GPU left on 2026-10-22) |
| nabinkoirala467 | nabinkoirala467 | in use by the user's other project (not for BreaKHis) |

## Code repository

GitHub: https://github.com/nabinkoirala452/breakhis_classification (public; first push 2026-10-22, commit b93c453).
The local folder is being renamed to `breakhis_classification`.

## Data and outputs

- Kaggle inputs: `nabinkoirala452/articafts` (binary folds/samples), `nabinkoirala452/articafts-notebook1-mc`
  (8-class), `nabinkoirala452/code-files`, `nabinkoirala452/code-files-mc`,
  `nabinkoirala452/zoomoe-results-binary-multiclass`, images `ambarish/breakhis`.
- Local copies of results: [VLM/results/vlm_finetune/](VLM/results/vlm_finetune/), [VLM/results/zoommoe_vlm/](VLM/results/zoommoe_vlm/), [VLM/results/zoommoe_ablation_binary/](VLM/results/zoommoe_ablation_binary/), [VLM/results/nb03_allfolds/](VLM/results/nb03_allfolds/), [VLM/results/baselines_resnet50/](VLM/results/baselines_resnet50/) (CSVs + log; the ~9 GB of checkpoints stay on Kaggle),
  [zoommoe_results.csv](zoommoe_results.csv), [zoommoe_multiclass_results.csv](zoommoe_multiclass_results.csv) (ZoomMoE + ResNet-50 per fold).

## Run costs

- One ZoomMoE + ResNet-50 fold: 40–50 min on one T4 in notebook 02 (disk loading); ~35 min in exp 07
  (RAM + GPU augmentation). ZoomMoE + ViT-B/32: ~40 min per fold.
- 05 per fold: QuiltNet/PLIP 2.4–3.5 min, BiomedCLIP 5.5–6.2, CLIP 7.5–8.3, SigLIP 7.2–8.7 (one T4);
  total 275 GPU-min = 2.41 h wall on T4×2; best val epoch mostly 1–4; early stopping in 45 of 50 runs.
- 04: CLIP-type embeddings ~1–2 min per model; generative models 11–82 min per task.

## Input verification for 08 and 09 (2026-10-22)

- Both scripts assert on Kaggle that every fold matches notebook 02's printed audit: sizes and patients;
  for 8-class also the test class counts per fold.
- 08 also asserts that each notebook-02 checkpoint (`notebook2-articafts-breakhis[-mc]`) reproduces that
  fold's notebook-02 TEST metrics (max abs diff < 0.01) before running any test.
- Model code: binary gets the same one-line AMP patch as 07; the 8-class `code-files-mc/zoommoe_model.py`
  already contains the fix (checked: exactly once).
- Dataset access checked from both accounts before pushing.

## Input verification for 07 (2026-10-21)

Downloaded from Kaggle and checked, not assumed:
- `code-files/breakhis_common.py` is identical to [Binary Class/code_files/breakhis_common.py](Binary%20Class/code_files/breakhis_common.py).
- `code-files/zoommoe_model.py` differs from the model the binary notebook 02 trained, which was written by
  its own `%%writefile` cell, in exactly one line: the fp16 `scatter` dtype fix. The unpatched line crashes
  under AMP. 07 applies that single-line patch at load time; the patched file was diffed equal to the
  notebook's model.
- `zoomoe-results-binary-multiclass` CSVs are identical to the local copies and to notebook 02's printed
  per-fold `TEST:` lines.
- `samples_pool.pkl` / `folds_all.npy` could not be unpickled locally (NumPy 2 pickles), so 07 asserts on
  Kaggle that every fold matches notebook 02's printed audit: sizes, patients, test malignant fraction.
  It also asserts the full model's trainable parameter count (48,538,906).
- Every training setting was compared line by line with notebook 02 and matches: seed, batch, 8/15/8
  epochs, learning rates, AdamW wd, OneCycle, clip 5, GRL ramp, scale dropout, freeze, normalisation,
  evaluation.

## Code review of the local code files (2026-10-21)

Local `Binary Class/code_files/*` and `MultiClass/code_files_mc/*` are byte-identical (ignoring line endings)
to the Kaggle datasets `code-files` / `code-files-mc`. Those datasets are what notebooks 01 and 03 (both tasks)
and the 8-class notebook 02 loaded. The binary notebook 02 alone used its own `%%writefile` model.
Binary and 8-class versions differ only in label handling (float/BCE vs long/CE, 1 vs 8 outputs, the
8-class AUC branch).

- **Bug: binary `zoommoe_model.py` line 104** lacks `.to(logits.dtype)` on the gate scatter, so it crashes under
  fp16 autocast. The 8-class file and the binary notebook's own copy have the fix. Results are unaffected, because
  training used the fixed copy. **Fixed locally 2026-10-21**: the local file now equals the notebook-02 model.
  The Kaggle `code-files` dataset still has the old line; exp 07 patches it at load time.
- Not bugs, but worth knowing (they shaped the reported results; changing them would change ZoomMoE):
  - `freeze_until` only sets `requires_grad`; frozen ResNet stages stay in train mode, so their BatchNorm
    running statistics still update in stages 0–1.
  - The backbone also runs on zero-filled slots of missing magnifications. The state update and the losses
    use the mask, but the routing statistics (load balance, I(E;M), expert-usage heatmaps) include those slots.
  - `make_image_dataset` uses one `random.Random(seed)`, copied into every DataLoader worker, so workers repeat
    the same candidate draws; image augmentation itself is unaffected.
  - Notebook 01's table column "Image Acc" is sample-level accuracy; name it that in the paper.
- Checked and correct: patient id = full biopsy prefix (82 patients); patient-disjoint StratifiedGroupKFold +
  patient-disjoint val; eval candidate `cands[i % len]`; scale dropout keeps ≥ 1 magnification; patient-safe
  InfoNCE and SupCon masks; ordinal, disentangle and adversarial losses use the mask; PRR = mean per-patient
  accuracy; 8-class OVR AUC handles absent classes.

## Incidents

- 07 first push (Kaggle user nabinkoirala452, 2026-10-21 22:40) failed in the smoke test after ~4 min:
  the code-files `zoommoe_model.py` lacks the AMP fix (see above). Relaunched on nabinkoirala45 with the fix.

- 05 first attempt: one GPU thread deadlocked when forking DataLoader workers; the run hit the 12 h limit.
  Fixed by GPU-side augmentation with no worker processes, plus a 30-min stall watchdog (also in 06, 07).
