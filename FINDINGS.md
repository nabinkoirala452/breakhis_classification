# Findings (interpretation of [RESULTS.md](RESULTS.md))

1. **Zero-shot VLMs are near chance.** The generative VLMs give the same answer for every image (binary:
   always "malignant"; 8-class: always "ductal" or "papillary"), so their accuracy equals the majority-class
   baseline. Blank-image calibration helps binary macro-F1 a little but breaks 8-class, so the problem is
   missing knowledge, not only answer bias. Pathology contrastive models beat general ones zero-shot on
   8-class (QuiltNet 0.305 vs CLIP 0.084 Acc), but binary zero-shot AUC is near 0.5 for every model.
2. **Full fine-tuning does not beat the linear probe.** It is about equal on binary and slightly worse on
   8-class. With 82 patients, end-to-end fine-tuning of an 86M-parameter ViT overfits fast.
3. **ZoomMoE + ResNet-50 is best on most metrics**:
   - binary: Acc, PRR, macro-F1;
   - 8-class: Acc, PRR, AUC.

   Exceptions: binary AUC (fine-tuned BiomedCLIP 0.979 vs 0.973), 8-class macro-F1 (fine-tuned BiomedCLIP
   0.270 / QuiltNet 0.264 vs 0.261) and 8-class Acc (QuiltNet linear probe 0.498 vs 0.496). Every margin is
   within one fold-std, so with 5 folds none is significant. Write "comparable or better", not "significantly
   better".
4. **Binary fold 4 is ZoomMoE's weak spot**: Acc 0.642 while every VLM gets 0.69–0.78, yet its AUC there is
   0.930. The ranking is fine and the 0.5 threshold is miscalibrated on that fold. Likely cause: class-prior
   shift. Fold 4's test set is 50.1% malignant, while its training set is 77.4% (the other folds' test sets
   are 72–79%). Exp 07 tests a val-tuned threshold.
5. **Every method is close to the majority baseline on 8-class accuracy** (0.443). Macro-F1 and AUC are the
   informative 8-class metrics.
6. **ZoomMoE's design premises hold on fold 0** (notebook 03): it is order-sensitive, zooming helps AUC, and
   the experts specialise by magnification. **But patient identity is linearly decodable** from the fused
   state (12–14× chance), which is a limitation to state.
7. **On 8-class, more magnifications improve AUC but not macro-F1** (fold 0: best macro-F1 with 2 mags).
   Worth a sentence in the paper.
8. Per magnification (fine-tuned VLMs, image level), no magnification is clearly best; binary peaks at
   100×/200× for most models.
9. **ZoomMoE + QuiltNet is the best 8-class model**: Acc 0.525, PRR 0.567 and AUC 0.747, the best of every
   method tried; macro-F1 0.253 is not. Against fine-tuned QuiltNet, the same encoder without ZoomMoE, it adds
   +0.052 Acc, +0.072 PRR (4/5 folds) and +0.041 AUC, but −0.011 macro-F1. So the ZoomMoE design adds value
   on top of a pathology VLM, but the gain comes from the frequent classes, not the rare ones. PLIP as
   backbone is worse than ResNet-50 on 8-class.
10. **For binary, the VLM backbone does not help**: ZoomMoE + ResNet-50 stays best (AUC 0.973 vs 0.922
    QuiltNet / 0.943 PLIP). Paper framing: the backbone choice is task-dependent; pathology pretraining
    pays off on the fine-grained subtype task.
11. **Binary ablation (exp 07):**
    - The pipeline reproduces notebook 02: the full rerun gives 0.893 / 0.971 vs the original 0.891 / 0.973.
    - **Routing on magnification only is better than the full model** (+0.022 Acc, +0.030 macro-F1, +0.015 AUC;
      AUC higher in 4/5 folds). So the zoom-state feedback into the router (h_{k-1}) does not help binary,
      which contradicts one design claim unless 8-class shows otherwise.
    - Removing the MoE leaves accuracy unchanged and costs 0.014 AUC, with 43% fewer parameters
      (27.5 M vs 48.5 M).
    - SupCon is the most useful loss for ranking (−0.046 AUC without it). The contrastive losses help a little
      (AUC lower without them in 5/5 folds, −0.018). The ordinal loss helps thresholded metrics (−0.017 Acc
      without it), not AUC.
    - All differences are within one fold-std. Binary is near ceiling, so it cannot separate the parts well;
      the 8-class ablation is needed.
12. **The val-tuned threshold does not fix fold 4** (0.667 → 0.693) and does not help on average. The val
    split has only 10 patients, so the chosen threshold swings between 0.05 and 0.95 across folds. The
    fold-4 drop is a class-prior shift that this val split cannot detect. Keep the 0.5 threshold and report
    the shift as a limitation.
13. **8-class test folds miss whole classes** (from notebook 02's audit): fold 0 has no adenosis; fold 1 has no
    adenosis, phyllodes or tubular adenoma; fold 2 no lobular; fold 3 no mucinous; fold 4 no adenosis,
    fibroadenoma or phyllodes. With only 82 patients (some classes have ~7), patient-disjoint stratification
    cannot place every class in every fold. Per-fold 8-class macro-F1 therefore averages over different
    class sets, which is part of why it is low and noisy. State this in the paper; pooled-over-folds
    per-class metrics are the fairer view (TODO: statistics task).
14. **Simple same-backbone fusion beats ZoomMoE + ResNet-50 (exp 09), on both tasks.** ResNet-50 trained with
    ZoomMoE's exact recipe and plain fusion (mean / concat / magnification-averaged single image) is higher on
    nearly every metric, with half the parameters (23.5 M vs 48.5 M):
    - binary: mean_fusion 0.910 Acc / 0.894 macro-F1 vs 0.891 / 0.866; late_avg AUC 0.984 vs 0.973;
    - 8-class: mean_fusion AUC 0.787 and macro-F1 0.281 vs 0.740 / 0.261; late_avg Acc 0.529 vs 0.496.

    ZoomMoE wins only 1–3 of 5 folds in every paired comparison. Mean fusion also beats ZoomMoE + QuiltNet on
    8-class AUC (0.787 vs 0.747) and macro-F1 (0.281 vs 0.253), and ties it on Acc/PRR. Differences are within
    fold noise, but they all point the same way. **The current ZoomMoE cannot be claimed to beat simple
    multi-scale fusion.**
    - The pipeline is not the cause: the exp-07 ZoomMoE full rerun on the same RAM/GPU pipeline reproduced
      notebook 02 (0.893 / 0.971).
    - One difference favours the baselines: ZoomMoE's stage 0 (8 epochs) is unsupervised, while the
      baselines train supervised in all 31 epochs. A ZoomMoE run with a supervised stage 0 would test whether
      that explains the gap.
    - The VLM fine-tunes (exp 05, 0.47 8-class Acc) used a different recipe (image-level, ≤ 10 epochs). The
      ZoomMoE recipe itself, with pooled pairing and 31 staged epochs, is strong even for a plain ResNet-50.
15. **Across all 5 folds the ZoomMoE premises mostly do not hold (exp 08).** Fold 0, the one notebook 03 used,
    was the favourable fold.
    - **Order sensitivity fails**: mean AUC gap −0.003 (binary) and −0.006 (8-class); it passes in 1/5 folds
      per task. The progressive zoom does not use the 40×→400× order.
    - **Zoom gain is real but saturates**: AUC 0.925 → 0.962 (binary) and 0.702 → 0.739 (8-class) from 1 to 4
      magnifications; 8-class plateaus after 2–3, and macro-F1 peaks at 2 magnifications (0.278 → 0.250 at 4).
    - Experts do specialise by magnification (I(E;M) ≈ 0.8 bits, 5/5 folds). The ordinal z_spec structure holds
      in 5/5, as expected, since a loss enforces it.
    - **The patient-identity probe fails in 10/10 folds** (11–14× chance).
    - Random-label control: mean AUC 0.50–0.52, so there is no leakage signal on average. Single folds swing
      0.35–0.87, because a 16-patient test set makes sample-level AUC follow a few patients.
16. **Implication for the paper:** the multi-scale gain is real (zoom gain, and fusion beats single
    magnification), but it comes from combining magnifications, not from ZoomMoE's specific machinery
    (MoE routing, ordered progressive zoom, auxiliary losses). That matches exp 07 (removing the MoE costs
    nothing in accuracy) and exp 08 (no order sensitivity).
