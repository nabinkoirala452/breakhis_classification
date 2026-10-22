# BreaKHis multi-scale breast histopathology classification

Benign vs malignant (binary) and 8-subtype classification on BreaKHis under a **patient-disjoint**
5-fold protocol. The project compares **ZoomMoE**, a magnification-aware mixture-of-experts model with
progressive zoom, with CNN baselines and small vision-language models (VLMs).

- Results: [RESULTS.md](RESULTS.md) (tables only)
- What they mean: [FINDINGS.md](FINDINGS.md)
- Methods and settings: [PROTOCOL.md](PROTOCOL.md)
- Run log, accounts, costs: [EXPERIMENTS.md](EXPERIMENTS.md)
- Next steps: [TODO.md](TODO.md)

## Repository structure

```
├── notebooks/            the original pipeline, one folder per task
│   ├── binary/           01 protocol + baselines · 02 ZoomMoE training · 03 analysis / falsification
│   └── multiclass/       same three notebooks for the 8-class task
├── src/                  shared code imported by the notebooks and experiments
│   ├── binary/           breakhis_common.py (data, folds, metrics) · zoommoe_model.py
│   └── multiclass/
├── experiments/          follow-up experiments 04–09 (one Kaggle script each, "# %% [Cell N]" blocks)
│   └── kaggle/           build_notebook.py + the exact notebooks pushed to Kaggle (runs/)
├── results/              outputs downloaded from Kaggle, one folder per experiment
└── docs/                 literature review
```

## Experiments

All experiments run on Kaggle (GPU T4 ×2) on the same patient-disjoint folds from notebook 01.

| # | What | Code | Kaggle notebook | Results |
|---|---|---|---|---|
| 01 | Protocol repair, patient-disjoint folds, frozen-VGG16 baselines (binary) | [notebooks/binary/01_protocol_and_baselines.ipynb](notebooks/binary/01_protocol_and_baselines.ipynb) | [breakhis-exp-research](https://www.kaggle.com/code/nabinkoirala452/breakhis-exp-research) | in notebook |
| 01 | same, 8-class | [notebooks/multiclass/01_protocol_and_baselines.ipynb](notebooks/multiclass/01_protocol_and_baselines.ipynb) | [mc-01-protocol-and-baselines-ipynb](https://www.kaggle.com/code/nabinkoirala452/mc-01-protocol-and-baselines-ipynb) | in notebook |
| 02 | ZoomMoE + ResNet-50 training, 5 folds (binary) | [notebooks/binary/02_zoommoe_train.ipynb](notebooks/binary/02_zoommoe_train.ipynb) | [02-zoommoe-train-ipynb](https://www.kaggle.com/code/nabinkoirala452/02-zoommoe-train-ipynb) | [results/02_zoommoe_resnet50/](results/02_zoommoe_resnet50/) |
| 02 | same, 8-class | [notebooks/multiclass/02_zoommoe_train.ipynb](notebooks/multiclass/02_zoommoe_train.ipynb) | [mc-02-zoommoe-train-ipynb](https://www.kaggle.com/code/nabinkoirala452/mc-02-zoommoe-train-ipynb) | [results/02_zoommoe_resnet50/](results/02_zoommoe_resnet50/) |
| 03 | ZoomMoE analysis / falsification tests, fold 0 (binary) | [notebooks/binary/03_analysis_and_falsification.ipynb](notebooks/binary/03_analysis_and_falsification.ipynb) | [breakhis-03-analysis-and-falsification](https://www.kaggle.com/code/nabinkoirala452/breakhis-03-analysis-and-falsification) | in notebook |
| 03 | same, 8-class | [notebooks/multiclass/03_analysis_and_falsification.ipynb](notebooks/multiclass/03_analysis_and_falsification.ipynb) | [mc-breakhis-03-analysis-and-falsification](https://www.kaggle.com/code/nabinkoirala452/mc-breakhis-03-analysis-and-falsification) | in notebook |
| 04 | VLMs zero-shot, linear probe, generative zero-shot | [experiments/04_vlm_zeroshot_linear_probe.py](experiments/04_vlm_zeroshot_linear_probe.py) | [vlm-breakhis-classification](https://www.kaggle.com/code/nabinkoirala452/vlm-breakhis-classification) | [results/04_vlm_zeroshot_linear_probe/](results/04_vlm_zeroshot_linear_probe/) |
| 05 | Full fine-tuning of 5 VLM image encoders | [experiments/05_vlm_finetune.py](experiments/05_vlm_finetune.py) | [vlm-finetune](https://www.kaggle.com/code/nabinkoirala452/vlm-finetune) | [results/05_vlm_finetune/](results/05_vlm_finetune/) |
| 06 | ZoomMoE with VLM backbones (QuiltNet, PLIP) | [experiments/06_zoommoe_vlm_backbone.py](experiments/06_zoommoe_vlm_backbone.py) | [zoommoe-vlm](https://www.kaggle.com/code/nabinkoirala53/zoommoe-vlm) | [results/06_zoommoe_vlm_backbone/](results/06_zoommoe_vlm_backbone/) |
| 07 | ZoomMoE training-time ablations (binary) | [experiments/07_zoommoe_ablation.py](experiments/07_zoommoe_ablation.py) | [zoommoe-ablation-binary](https://www.kaggle.com/code/nabinkoirala45/zoommoe-ablation-binary) | [results/07_zoommoe_ablation_binary/](results/07_zoommoe_ablation_binary/) |
| 08 | Notebook-03 tests on all 5 folds, both tasks | [experiments/08_zoommoe_tests_all_folds.py](experiments/08_zoommoe_tests_all_folds.py) | [zoommoe-nb03-allfolds](https://www.kaggle.com/code/nabinkoirala452/zoommoe-nb03-allfolds) | [results/08_zoommoe_tests_all_folds/](results/08_zoommoe_tests_all_folds/) |
| 09 | Same-backbone ResNet-50 fusion baselines | [experiments/09_resnet50_fusion_baselines.py](experiments/09_resnet50_fusion_baselines.py) | [zoommoe-baselines-resnet50](https://www.kaggle.com/code/nabinkoirala53/zoommoe-baselines-resnet50) | [results/09_resnet50_fusion_baselines/](results/09_resnet50_fusion_baselines/) |

The Kaggle notebooks are private; the links open for their owner accounts.
Experiments 07–09 were pushed through the Kaggle API from [experiments/kaggle/runs/](experiments/kaggle/runs/)
with [experiments/kaggle/build_notebook.py](experiments/kaggle/build_notebook.py).

## Kaggle datasets used

| Dataset | Contents |
|---|---|
| [ambarish/breakhis](https://www.kaggle.com/datasets/ambarish/breakhis) | BreaKHis v1 images (7,909, 82 patients) |
| [nabinkoirala452/articafts](https://www.kaggle.com/datasets/nabinkoirala452/articafts) | binary: `samples_all.pkl`, `samples_pool.pkl`, `folds_all.npy` (notebook 01 output) |
| [nabinkoirala452/articafts-notebook1-mc](https://www.kaggle.com/datasets/nabinkoirala452/articafts-notebook1-mc) | same, 8-class |
| [nabinkoirala452/code-files](https://www.kaggle.com/datasets/nabinkoirala452/code-files) | `src/binary/*` as uploaded (its `zoommoe_model.py` lacks the one-line AMP fix in `src/binary/`) |
| [nabinkoirala452/code-files-mc](https://www.kaggle.com/datasets/nabinkoirala452/code-files-mc) | `src/multiclass/*` |
| [nabinkoirala452/notebook2-articafts-breakhis](https://www.kaggle.com/datasets/nabinkoirala452/notebook2-articafts-breakhis) | binary ZoomMoE checkpoints + test predictions, 5 folds (notebook 02 output) |
| [nabinkoirala452/notebook2-articafts-breakhis-mc](https://www.kaggle.com/datasets/nabinkoirala452/notebook2-articafts-breakhis-mc) | same, 8-class |
| [nabinkoirala452/zoomoe-results-binary-multiclass](https://www.kaggle.com/datasets/nabinkoirala452/zoomoe-results-binary-multiclass) | ZoomMoE per-fold result CSVs (= `results/02_zoommoe_resnet50/`) |

## Running an experiment

1. Create a Kaggle notebook with GPU T4 ×2 and Internet on, and attach the datasets listed at the top of
   the script.
2. Paste the script (all cells, or one `# %% [Cell N]` block per cell) and use **Save & Run All**.
3. Each script verifies its inputs (fold sizes and patients against notebook 02, checkpoints reproducing
   their scores), saves every finished fold, and stops inside a time budget below Kaggle's 12-hour limit.
