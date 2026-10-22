# Protocol (methods details for the paper)

## Evaluation (all experiments)

- 82 patients; 5-fold patient-disjoint CV (`StratifiedGroupKFold`, seed 42) with a patient-disjoint val split
  (15%) inside each training fold: `folds_all.npy` from notebook 01. Every experiment uses these folds.
- Unit: the multi-scale sample (patient, image id) with up to 4 magnifications; image probabilities are
  averaged over the magnifications a sample has. Image-level per-magnification scores are reported as well.
- Metrics as in `breakhis_common.evaluate`: accuracy, patient recognition rate (mean over patients of the
  per-patient accuracy), macro-F1, ROC-AUC (one-vs-rest for 8-class). Binary threshold 0.5.
- Reported as mean ± std over the 5 test folds.

## 04 — zero-shot, linear probe, generative VLMs

Script: [experiments/04_vlm_zeroshot_linear_probe.py](experiments/04_vlm_zeroshot_linear_probe.py)

- Contrastive (CLIP ViT-B/16, SigLIP B/16, PLIP, QuiltNet B/32, BiomedCLIP): prompt ensemble of 5 templates
  × 2–3 class descriptions; softmax over logit-scaled cosine similarities.
- Linear probe: logistic regression (class-balanced) on frozen, standardised image embeddings, trained on the
  fold's train images; C ∈ {0.001, 0.01, 0.1, 1} chosen on val sample-level macro-F1.
- Generative (Qwen2-VL-2B, Qwen2.5-VL-3B, SmolVLM2-2.2B, SmolVLM-500M): multiple-choice prompt that states the
  magnification; class probability = softmax over the next-token logits of the option letters.
  "Calibrated" = contextual calibration (Zhao et al., ICML 2021) with white/grey/black content-free images.
  Qwen max 448×448 px; SmolVLM2 used its default image tiling (the splitting switch did not apply).

## 05 — full fine-tuning

Script: [experiments/05_vlm_finetune.py](experiments/05_vlm_finetune.py)

- Whole image encoder + linear head; the head is initialised from the model's own prompt-ensembled text
  embeddings (zero-shot head init, Wortsman et al., CVPR 2022).
- AdamW, lr 1e-5 encoder / 1e-3 head, weight decay 0.05, 1 warm-up epoch + cosine, ≤ 10 epochs, early stopping
  (patience 3) on val sample-level macro-F1; class-balanced cross-entropy, label smoothing 0.1, batch 32, fp16.
- Input and augmentation as ZoomMoE: 224×224 squash resize, horizontal/vertical flips, rotation ±90° (p = 0.5),
  colour jitter 0.15/0.15/0.15/0.03; normalisation with each VLM's own mean/std. Augmentation runs on the GPU.

## 06 — ZoomMoE with a VLM backbone

Script: [experiments/06_zoommoe_vlm_backbone.py](experiments/06_zoommoe_vlm_backbone.py)

- [src/binary/zoommoe_model.py](src/binary/zoommoe_model.py) unchanged; only `MoEBackbone` is swapped. The ViT (QuiltNet / PLIP, ViT-B/32 → 7×7 patch
  grid, the same as ResNet-50's 7×7 map) gets token-MoE blocks after block round(2L/3) and block L.
  The experts' output layers are zero-initialised, so each block starts as the identity.
- Same 3 stages (8/15/8 epochs), lr 3e-4/3e-4/3e-5, batch 8, scale dropout 0.15, pooled pairing.
  Pretrained ViT weights train at 0.1× the stage lr. `freeze_until("layer2")` freezes the patch embedding
  and the first half of the blocks.

## 07 — ZoomMoE ablations (binary)

Script: [experiments/07_zoommoe_ablation.py](experiments/07_zoommoe_ablation.py)

- Notebook 02 recipe exactly, ResNet-50, 5 folds; one change at a time, with the overrides from notebook 02's
  `ABLATIONS` dict: `no_moe` (moe_stages=()), `moe_mag_only` (cond_state_dim=0), `no_contrastive` (all
  auxiliary contrastive weights 0), `full` rerun, then `no_supcon`, `no_ordinal` if time allows.
- Also saves val predictions; the val-tuned threshold (maximising val macro-F1) is evaluated on test.

## 08 — notebook-03 tests on all folds

Script: [experiments/08_zoommoe_tests_all_folds.py](experiments/08_zoommoe_tests_all_folds.py)

- Notebook 03's code, unchanged, looped over the 5 folds × 2 tasks with the notebook-02 checkpoints:
  order permutation (6 random orders + reversed), patient probe on h and z_inv (logistic regression,
  within-patient 50/50 split, patients with ≥ 6 samples), I(E;M) with a 200× batch bootstrap, zoom-gain
  curve from the step predictions, ordinal z_spec distances, and the random-label control (labels shuffled
  within the training fold, 3 epochs, fp32, as in notebook 03 test 5).
- Pass criteria as in notebook 03: AUC gap ≥ 0.01; probe < 5× chance; I(E;M) ≥ 0.05 bits; AUC lift ≥ 0.01;
  monotone ordinal distances; random-label AUC ≤ 0.60.

## 09 — same-backbone baselines

Script: [experiments/09_resnet50_fusion_baselines.py](experiments/09_resnet50_fusion_baselines.py)

- ImageNet ResNet-50 with ZoomMoE's exact training recipe (pooled pairing, scale dropout 0.15, augmentation,
  batch 8, 8/15/8 epochs, lr 3e-4/3e-4/3e-5, freeze to layer2 in the first two stages, AdamW 1e-4,
  OneCycle 0.3, clip 5, fp16). All three stages are supervised (ZoomMoE's stage 0 is unsupervised).
- `late_avg`: shared head on every magnification image, loss on each image; prediction = mean probability
  over the sample's magnifications. `mean_fusion`: masked mean of the 4 pooled features → head.
  `concat_fusion`: concatenated pooled features (missing = 0) + mask bits → head. Head = dropout 0.3 + linear.
- Binary: 1 logit + BCE; 8-class: CE. No class weighting, as in ZoomMoE.

## Known protocol differences (state in the paper)

- The VLM baselines score the image-id-paired table (`samples_all`); ZoomMoE scores the pooled-pairing table
  (`samples_pool`, deterministic candidate `cands[i % len]`). Same samples, patients and folds, but the
  image drawn per magnification can differ.
- Runs 05–07 load images from RAM and augment on the GPU; the ops are the same as `breakhis_common`, the
  random streams differ.
