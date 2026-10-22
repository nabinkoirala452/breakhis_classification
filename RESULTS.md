# Results

Sample-level metrics, mean ± std over the 5 patient-disjoint test folds (how they are computed: [PROTOCOL.md](PROTOCOL.md)).
PRR = patient recognition rate.

## Binary (benign vs malignant)

| Setting | Model | Acc | PRR | Macro-F1 | AUC |
|---|---|---|---|---|---|
| baseline | majority class (always malignant) | 0.704 | 0.708 | 0.411 | 0.500 |
| zero-shot | CLIP ViT-B/16 | 0.323 ± 0.113 | 0.314 ± 0.041 | 0.274 ± 0.072 | 0.481 ± 0.129 |
| zero-shot | SigLIP B/16 | 0.505 ± 0.110 | 0.515 ± 0.115 | 0.476 ± 0.125 | 0.494 ± 0.162 |
| zero-shot | PLIP | 0.637 ± 0.078 | 0.639 ± 0.071 | 0.465 ± 0.081 | 0.473 ± 0.160 |
| zero-shot | QuiltNet B/32 | 0.639 ± 0.067 | 0.630 ± 0.066 | 0.538 ± 0.092 | 0.566 ± 0.149 |
| zero-shot | BiomedCLIP | 0.538 ± 0.101 | 0.544 ± 0.089 | 0.514 ± 0.101 | 0.580 ± 0.107 |
| zero-shot gen. | Qwen2-VL-2B | 0.704 ± 0.116 | 0.708 ± 0.044 | 0.411 ± 0.044 | 0.407 ± 0.145 |
| zero-shot gen. | Qwen2.5-VL-3B (calibrated) | 0.566 ± 0.053 | 0.574 ± 0.067 | 0.525 ± 0.064 | 0.565 ± 0.059 |
| zero-shot gen. | SmolVLM2-2.2B (calibrated) | 0.703 ± 0.089 | 0.716 ± 0.040 | 0.501 ± 0.063 | 0.648 ± 0.102 |
| zero-shot gen. | SmolVLM-500M (calibrated) | 0.624 ± 0.116 | 0.642 ± 0.084 | 0.454 ± 0.074 | 0.448 ± 0.151 |
| linear probe | CLIP ViT-B/16 | 0.853 ± 0.087 | 0.856 ± 0.035 | 0.823 ± 0.092 | 0.929 ± 0.055 |
| linear probe | SigLIP B/16 | 0.862 ± 0.096 | 0.859 ± 0.049 | 0.834 ± 0.099 | 0.948 ± 0.033 |
| linear probe | PLIP | 0.884 ± 0.085 | 0.885 ± 0.034 | 0.861 ± 0.085 | 0.952 ± 0.026 |
| linear probe | QuiltNet B/32 | 0.869 ± 0.093 | 0.880 ± 0.052 | 0.846 ± 0.096 | 0.950 ± 0.029 |
| linear probe | BiomedCLIP | 0.847 ± 0.076 | 0.852 ± 0.044 | 0.819 ± 0.078 | 0.931 ± 0.050 |
| fine-tuned | CLIP ViT-B/16 | 0.862 ± 0.103 | 0.872 ± 0.049 | 0.833 ± 0.110 | 0.945 ± 0.078 |
| fine-tuned | SigLIP B/16 | 0.876 ± 0.086 | 0.885 ± 0.030 | 0.850 ± 0.088 | 0.923 ± 0.082 |
| fine-tuned | PLIP | 0.865 ± 0.101 | 0.882 ± 0.046 | 0.837 ± 0.107 | 0.949 ± 0.071 |
| fine-tuned | QuiltNet B/32 | 0.871 ± 0.073 | 0.885 ± 0.034 | 0.850 ± 0.073 | 0.934 ± 0.089 |
| fine-tuned | BiomedCLIP | 0.875 ± 0.077 | 0.891 ± 0.029 | 0.854 ± 0.079 | 0.979 ± 0.013 |
| same-backbone (exp 09) | ResNet-50 late_avg | 0.899 ± 0.089 | 0.905 ± 0.045 | 0.879 ± 0.092 | 0.984 ± 0.018 |
| same-backbone (exp 09) | ResNet-50 mean_fusion | 0.910 ± 0.087 | 0.916 ± 0.044 | 0.894 ± 0.090 | 0.975 ± 0.036 |
| same-backbone (exp 09) | ResNet-50 concat_fusion | 0.908 ± 0.095 | 0.918 ± 0.054 | 0.892 ± 0.102 | 0.976 ± 0.034 |
| ours | ZoomMoE + ResNet-50 | 0.891 ± 0.140 | 0.902 ± 0.086 | 0.866 ± 0.157 | 0.973 ± 0.027 |
| ours | ZoomMoE + QuiltNet B/32 (exp 06) | 0.879 ± 0.072 | 0.889 ± 0.030 | 0.853 ± 0.072 | 0.922 ± 0.082 |
| ours | ZoomMoE + PLIP (exp 06) | 0.882 ± 0.093 | 0.889 ± 0.039 | 0.855 ± 0.093 | 0.943 ± 0.065 |

## 8-class subtypes

| Setting | Model | Acc | PRR | Macro-F1 | AUC |
|---|---|---|---|---|---|
| baseline | majority class (always ductal) | 0.443 | 0.463 | 0.098 | 0.500 |
| zero-shot | CLIP ViT-B/16 | 0.084 ± 0.044 | 0.063 ± 0.038 | 0.036 ± 0.012 | 0.496 ± 0.074 |
| zero-shot | SigLIP B/16 | 0.090 ± 0.056 | 0.103 ± 0.074 | 0.067 ± 0.056 | 0.430 ± 0.133 |
| zero-shot | PLIP | 0.183 ± 0.050 | 0.157 ± 0.063 | 0.144 ± 0.087 | 0.656 ± 0.116 |
| zero-shot | QuiltNet B/32 | 0.305 ± 0.149 | 0.306 ± 0.140 | 0.188 ± 0.103 | 0.670 ± 0.094 |
| zero-shot | BiomedCLIP | 0.256 ± 0.040 | 0.247 ± 0.038 | 0.174 ± 0.036 | 0.632 ± 0.071 |
| zero-shot gen. | Qwen2-VL-2B / Qwen2.5-VL-3B / SmolVLM2 (raw) | 0.443 ± 0.102 | 0.463 ± 0.111 | 0.098 ± 0.031 | 0.37–0.59 |
| zero-shot gen. | all 4, calibrated | 0.05–0.08 | 0.04–0.08 | 0.02–0.03 | 0.39–0.54 |
| linear probe | CLIP ViT-B/16 | 0.461 ± 0.124 | 0.484 ± 0.100 | 0.212 ± 0.038 | 0.692 ± 0.016 |
| linear probe | SigLIP B/16 | 0.445 ± 0.103 | 0.477 ± 0.084 | 0.236 ± 0.020 | 0.715 ± 0.036 |
| linear probe | PLIP | 0.474 ± 0.147 | 0.493 ± 0.152 | 0.239 ± 0.031 | 0.738 ± 0.016 |
| linear probe | QuiltNet B/32 | 0.498 ± 0.101 | 0.526 ± 0.097 | 0.255 ± 0.026 | 0.732 ± 0.022 |
| linear probe | BiomedCLIP | 0.426 ± 0.103 | 0.451 ± 0.091 | 0.260 ± 0.037 | 0.710 ± 0.033 |
| fine-tuned | CLIP ViT-B/16 | 0.415 ± 0.153 | 0.441 ± 0.148 | 0.220 ± 0.053 | 0.667 ± 0.064 |
| fine-tuned | SigLIP B/16 | 0.429 ± 0.123 | 0.438 ± 0.115 | 0.242 ± 0.063 | 0.682 ± 0.047 |
| fine-tuned | PLIP | 0.441 ± 0.121 | 0.462 ± 0.120 | 0.258 ± 0.024 | 0.693 ± 0.070 |
| fine-tuned | QuiltNet B/32 | 0.473 ± 0.186 | 0.495 ± 0.197 | 0.264 ± 0.057 | 0.706 ± 0.042 |
| fine-tuned | BiomedCLIP | 0.409 ± 0.082 | 0.418 ± 0.080 | 0.270 ± 0.050 | 0.709 ± 0.031 |
| same-backbone (exp 09) | ResNet-50 late_avg | 0.529 ± 0.109 | 0.566 ± 0.087 | 0.249 ± 0.034 | 0.763 ± 0.028 |
| same-backbone (exp 09) | ResNet-50 mean_fusion | 0.516 ± 0.140 | 0.552 ± 0.120 | 0.281 ± 0.022 | 0.787 ± 0.058 |
| same-backbone (exp 09) | ResNet-50 concat_fusion | 0.517 ± 0.120 | 0.553 ± 0.096 | 0.273 ± 0.020 | 0.762 ± 0.036 |
| ours | ZoomMoE + ResNet-50 | 0.496 ± 0.119 | 0.532 ± 0.104 | 0.261 ± 0.075 | 0.740 ± 0.093 |
| ours | ZoomMoE + QuiltNet B/32 (exp 06) | 0.525 ± 0.105 | 0.567 ± 0.091 | 0.253 ± 0.046 | 0.747 ± 0.060 |
| ours | ZoomMoE + PLIP (exp 06) | 0.482 ± 0.112 | 0.515 ± 0.099 | 0.208 ± 0.018 | 0.725 ± 0.077 |

## Per fold: ZoomMoE vs fine-tuned VLMs

Folds won by ZoomMoE out of 5 (mean difference ZoomMoE − VLM):

| Task | vs | Acc | PRR | Macro-F1 | AUC |
|---|---|---|---|---|---|
| binary | BiomedCLIP | 3 (+0.016) | 3 (+0.011) | 3 (+0.012) | 2 (−0.006) |
| binary | QuiltNet | 3 (+0.020) | 3 (+0.017) | 3 (+0.016) | 5 (+0.039) |
| binary | PLIP | 4 (+0.025) | 4 (+0.021) | 4 (+0.028) | 4 (+0.024) |
| binary | SigLIP | 4 (+0.014) | 4 (+0.017) | 4 (+0.016) | 4 (+0.051) |
| binary | CLIP | 3 (+0.029) | 3 (+0.030) | 3 (+0.032) | 3 (+0.028) |
| 8-class | BiomedCLIP | 5 (+0.087) | 5 (+0.114) | 3 (−0.009) | 2 (+0.031) |
| 8-class | QuiltNet | 3 (+0.023) | 3 (+0.037) | 3 (−0.003) | 2 (+0.034) |
| 8-class | PLIP | 2 (+0.054) | 3 (+0.070) | 2 (+0.003) | 4 (+0.047) |
| 8-class | SigLIP | 3 (+0.067) | 4 (+0.094) | 3 (+0.019) | 3 (+0.058) |
| 8-class | CLIP | 2 (+0.081) | 2 (+0.091) | 3 (+0.041) | 3 (+0.073) |

Binary accuracy per fold:

| Fold | ZoomMoE | BiomedCLIP | CLIP | PLIP | QuiltNet | SigLIP |
|---|---|---|---|---|---|---|
| 0 | 0.920 | 0.928 | 0.872 | 0.894 | 0.896 | 0.876 |
| 1 | **0.977** | 0.877 | 0.880 | 0.895 | 0.903 | 0.893 |
| 2 | 0.956 | 0.859 | **0.962** | 0.898 | 0.821 | 0.924 |
| 3 | 0.958 | 0.954 | 0.906 | 0.950 | **0.960** | 0.956 |
| 4 | **0.642** | 0.755 | 0.689 | 0.689 | **0.776** | 0.733 |

ZoomMoE binary AUC per fold: 0.981 / 1.000 / 0.988 / 0.967 / 0.930.

## Fine-tuned VLMs: image-level accuracy per magnification

| Task | Model | 40× | 100× | 200× | 400× |
|---|---|---|---|---|---|
| binary | QuiltNet B/32 | 0.861 | 0.878 | 0.862 | 0.857 |
| binary | PLIP | 0.850 | 0.856 | 0.879 | 0.866 |
| binary | BiomedCLIP | 0.869 | 0.859 | 0.868 | 0.864 |
| binary | CLIP ViT-B/16 | 0.850 | 0.863 | 0.861 | 0.850 |
| binary | SigLIP B/16 | 0.860 | 0.881 | 0.886 | 0.866 |
| 8-class | QuiltNet B/32 | 0.432 | 0.438 | 0.433 | 0.431 |
| 8-class | PLIP | 0.400 | 0.409 | 0.442 | 0.430 |
| 8-class | BiomedCLIP | 0.360 | 0.366 | 0.370 | 0.336 |
| 8-class | CLIP ViT-B/16 | 0.396 | 0.418 | 0.417 | 0.388 |
| 8-class | SigLIP B/16 | 0.421 | 0.399 | 0.384 | 0.365 |

## Same-backbone baselines (exp 09): ResNet-50, ZoomMoE's training recipe, simple fusion

| Task | Model | Acc | PRR | Macro-F1 | AUC | Params |
|---|---|---|---|---|---|---|
| binary | ResNet-50 late_avg (single image, mag-averaged) | 0.899 ± 0.089 | 0.905 ± 0.045 | 0.879 ± 0.092 | **0.984 ± 0.018** | 23.5 M |
| binary | ResNet-50 mean_fusion | **0.910 ± 0.087** | 0.916 ± 0.044 | **0.894 ± 0.090** | 0.975 ± 0.036 | 23.5 M |
| binary | ResNet-50 concat_fusion | 0.908 ± 0.095 | **0.918 ± 0.054** | 0.892 ± 0.102 | 0.976 ± 0.034 | 23.5 M |
| binary | ZoomMoE + ResNet-50 (notebook 02) | 0.891 ± 0.140 | 0.902 ± 0.086 | 0.866 ± 0.157 | 0.973 ± 0.027 | 48.5 M |
| 8-class | ResNet-50 late_avg (single image, mag-averaged) | **0.529 ± 0.109** | **0.566 ± 0.087** | 0.249 ± 0.034 | 0.763 ± 0.028 | 23.5 M |
| 8-class | ResNet-50 mean_fusion | 0.516 ± 0.140 | 0.552 ± 0.120 | **0.281 ± 0.022** | **0.787 ± 0.058** | 23.5 M |
| 8-class | ResNet-50 concat_fusion | 0.517 ± 0.120 | 0.553 ± 0.096 | 0.273 ± 0.020 | 0.762 ± 0.036 | 23.6 M |
| 8-class | ZoomMoE + ResNet-50 (notebook 02) | 0.496 ± 0.119 | 0.532 ± 0.104 | 0.261 ± 0.075 | 0.740 ± 0.093 | 48.5 M |
| 8-class | ZoomMoE + QuiltNet (exp 06) | 0.525 ± 0.105 | 0.567 ± 0.091 | 0.253 ± 0.046 | 0.747 ± 0.060 | 94.9 M |

ZoomMoE + ResNet-50 minus baseline (folds where ZoomMoE is higher, out of 5):

| Task | Baseline | Acc | PRR | Macro-F1 | AUC |
|---|---|---|---|---|---|
| binary | mean_fusion | −0.020 (2) | −0.014 (2) | −0.028 (2) | −0.002 (3) |
| binary | late_avg | −0.008 (2) | −0.003 (2) | −0.013 (2) | −0.011 (2) |
| binary | concat_fusion | −0.018 (2) | −0.016 (2) | −0.026 (2) | −0.003 (3) |
| 8-class | mean_fusion | −0.020 (3) | −0.020 (2) | −0.019 (1) | −0.047 (2) |
| 8-class | late_avg | −0.033 (2) | −0.034 (2) | +0.012 (2) | −0.023 (2) |
| 8-class | concat_fusion | −0.021 (2) | −0.021 (2) | −0.012 (1) | −0.022 (2) |

Accuracy per fold:

| Task | Model | f0 | f1 | f2 | f3 | f4 |
|---|---|---|---|---|---|---|
| binary | late_avg | 0.950 | 0.939 | 0.896 | 0.962 | 0.746 |
| binary | mean_fusion | 0.967 | 0.951 | 0.902 | 0.967 | 0.763 |
| binary | concat_fusion | 0.983 | 0.954 | 0.876 | 0.973 | 0.755 |
| binary | ZoomMoE + ResNet-50 | 0.920 | 0.977 | 0.956 | 0.958 | 0.642 |
| 8-class | late_avg | 0.620 | 0.671 | 0.429 | 0.462 | 0.461 |
| 8-class | mean_fusion | 0.647 | 0.653 | 0.315 | 0.479 | 0.487 |
| 8-class | concat_fusion | 0.632 | 0.643 | 0.359 | 0.473 | 0.480 |
| 8-class | ZoomMoE + ResNet-50 | 0.660 | 0.554 | 0.353 | 0.421 | 0.492 |

## ZoomMoE tests on all 5 folds (exp 08; notebook-03 code, notebook-02 checkpoints)

Each checkpoint reproduced its notebook-02 TEST metrics exactly (max abs diff 0.0000).

| Test (pass criterion) | Binary mean ± std | Binary folds passing | 8-class mean ± std | 8-class folds passing |
|---|---|---|---|---|
| Order permutation: AUC ordered − permuted (≥ 0.01) | −0.003 ± 0.016 | 1/5 | −0.006 ± 0.022 | 1/5 |
| Patient probe on h, × chance (< 5) | 11.6 ± 0.8 | 0/5 | 13.2 ± 0.8 | 0/5 |
| Expert specialisation I(E;M), bits (≥ 0.05) | 0.786 ± 0.161 | 5/5 | 0.817 ± 0.076 | 5/5 |
| Zoom gain: AUC lift 1 → 4 mags (≥ 0.01) | +0.036 ± 0.023 | 4/5 | +0.037 ± 0.077 | 3/5 |
| Ordinal z_spec structure (monotone) | — | 5/5 | — | 5/5 |
| Random-label control AUC (≤ 0.60) | 0.499 ± 0.228 | 4/5 | 0.515 ± 0.092 | 4/5 |

Per fold:

| Task | Fold | Test AUC | Order gap | Reversed gap | Probe h (×) | I(E;M) | Zoom lift | Random-label AUC |
|---|---|---|---|---|---|---|---|---|
| binary | 0 | 0.981 | +0.020 | +0.038 | 11.9 | 0.774 | +0.040 | 0.562 |
| binary | 1 | 1.000 | +0.002 | +0.004 | 12.2 | 0.858 | +0.002 | 0.873 |
| binary | 2 | 0.988 | −0.005 | −0.005 | 10.1 | 0.887 | +0.052 | 0.357 |
| binary | 3 | 0.967 | −0.008 | −0.012 | 11.9 | 0.900 | +0.027 | 0.354 |
| binary | 4 | 0.930 | −0.024 | −0.031 | 11.6 | 0.513 | +0.060 | 0.348 |
| 8-class | 0 | 0.885 | +0.022 | +0.024 | 13.8 | 0.686 | +0.068 | 0.475 |
| 8-class | 1 | 0.661 | +0.009 | +0.037 | 12.7 | 0.846 | −0.001 | 0.677 |
| 8-class | 2 | 0.725 | −0.029 | −0.041 | 14.1 | 0.883 | +0.089 | 0.469 |
| 8-class | 3 | 0.659 | −0.007 | −0.011 | 12.5 | 0.830 | −0.080 | 0.462 |
| 8-class | 4 | 0.769 | −0.025 | −0.033 | 12.7 | 0.838 | +0.107 | 0.491 |

Zoom-gain curve, mean over folds (ZoomMoE step predictions after k magnifications):

| Task | k | Acc | Macro-F1 | AUC |
|---|---|---|---|---|
| binary | 1 | 0.860 | 0.820 | 0.925 |
| binary | 2 | 0.883 | 0.855 | 0.950 |
| binary | 3 | 0.893 | 0.870 | 0.959 |
| binary | 4 | 0.890 | 0.865 | 0.962 |
| 8-class | 1 | 0.468 | 0.264 | 0.702 |
| 8-class | 2 | 0.491 | 0.278 | 0.731 |
| 8-class | 3 | 0.491 | 0.266 | 0.740 |
| 8-class | 4 | 0.488 | 0.250 | 0.739 |

## ZoomMoE inference-time tests (notebook 03, fold 0 only)

| Test | Binary | 8-class | Pass |
|---|---|---|---|
| Order permutation (AUC ordered − permuted mean) | +0.021 | +0.023 | yes |
| Zoom gain (AUC: 40× only → all 4 mags) | +0.040 | +0.068 | yes |
| Expert specialisation I(E;M), ceiling 2 bits [95% CI] | 0.774 [0.763, 0.791] | 0.686 [0.684, 0.704] | yes |
| Ordinal zoom structure on held-out data | holds | holds | yes |
| Patient-identity probe on fused state h (× chance) | 11.9× | 13.8× | **no** |

8-class, fold 0, by number of magnifications seen (1/2/3/4): macro-F1 0.406 / 0.472 / 0.388 / 0.346;
AUC 0.812 / 0.847 / 0.868 / 0.880.

## ZoomMoE with a VLM backbone: per fold (exp 06)

Folds won out of 5 (mean difference):

| Task | Comparison | Acc | PRR | Macro-F1 | AUC |
|---|---|---|---|---|---|
| binary | ZoomMoE+QuiltNet vs fine-tuned QuiltNet | 3 (+0.008) | 3 (+0.004) | 2 (+0.002) | 2 (−0.012) |
| binary | ZoomMoE+PLIP vs fine-tuned PLIP | 4 (+0.017) | 4 (+0.008) | 4 (+0.017) | 2 (−0.006) |
| 8-class | ZoomMoE+QuiltNet vs fine-tuned QuiltNet | 3 (+0.052) | 4 (+0.072) | 2 (−0.011) | 3 (+0.041) |
| 8-class | ZoomMoE+PLIP vs fine-tuned PLIP | 2 (+0.041) | 4 (+0.053) | 0 (−0.050) | 4 (+0.032) |
| binary | ZoomMoE+QuiltNet vs ZoomMoE+ResNet-50 | 1 (−0.011) | 2 (−0.013) | 1 (−0.013) | 1 (−0.051) |
| binary | ZoomMoE+PLIP vs ZoomMoE+ResNet-50 | 1 (−0.009) | 2 (−0.013) | 1 (−0.011) | 1 (−0.030) |
| 8-class | ZoomMoE+QuiltNet vs ZoomMoE+ResNet-50 | 4 (+0.029) | 3 (+0.035) | 2 (−0.008) | 3 (+0.007) |
| 8-class | ZoomMoE+PLIP vs ZoomMoE+ResNet-50 | 2 (−0.013) | 2 (−0.017) | 1 (−0.053) | 2 (−0.015) |

8-class accuracy per fold:

| Fold | ZoomMoE+ResNet-50 | ZoomMoE+QuiltNet | ZoomMoE+PLIP |
|---|---|---|---|
| 0 | **0.660** | 0.613 | 0.594 |
| 1 | 0.554 | **0.609** | 0.589 |
| 2 | 0.353 | **0.355** | 0.330 |
| 3 | 0.421 | **0.515** | 0.431 |
| 4 | 0.492 | **0.534** | 0.468 |

## ZoomMoE training-time ablations, binary (exp 07, ResNet-50, 5 folds)

| Variant | Acc | PRR | Macro-F1 | AUC |
|---|---|---|---|---|
| full (notebook 02) | 0.891 ± 0.140 | 0.902 ± 0.086 | 0.866 ± 0.157 | 0.973 ± 0.027 |
| full (rerun, same pipeline as the variants) | 0.893 ± 0.128 | 0.902 ± 0.083 | 0.871 ± 0.141 | 0.971 ± 0.040 |
| no MoE (`moe_stages=()`) | 0.892 ± 0.117 | 0.903 ± 0.069 | 0.867 ± 0.126 | 0.957 ± 0.050 |
| MoE routed on magnification only (`cond_state_dim=0`) | **0.915 ± 0.115** | **0.922 ± 0.067** | **0.901 ± 0.123** | **0.986 ± 0.022** |
| no contrastive losses | 0.879 ± 0.083 | 0.893 ± 0.043 | 0.856 ± 0.083 | 0.953 ± 0.043 |
| no SupCon | 0.887 ± 0.128 | 0.904 ± 0.075 | 0.860 ± 0.142 | 0.925 ± 0.088 |
| no ordinal loss | 0.876 ± 0.145 | 0.895 ± 0.090 | 0.848 ± 0.167 | 0.973 ± 0.035 |

Variant minus full rerun: folds where the variant is higher, out of 5 (mean difference):

| Variant | Acc | PRR | Macro-F1 | AUC |
|---|---|---|---|---|
| no MoE | 3 (−0.001) | 2 (+0.001) | 3 (−0.004) | 2 (−0.014) |
| MoE mag-only routing | 3 (+0.022) | 3 (+0.020) | 3 (+0.030) | 4 (+0.015) |
| no contrastive | 2 (−0.014) | 3 (−0.009) | 2 (−0.014) | 0 (−0.018) |
| no SupCon | 4 (−0.006) | 3 (+0.002) | 4 (−0.011) | 1 (−0.046) |
| no ordinal | 1 (−0.017) | 2 (−0.007) | 1 (−0.023) | 3 (+0.002) |

Accuracy per fold:

| Variant | f0 | f1 | f2 | f3 | f4 |
|---|---|---|---|---|---|
| full (rerun) | 0.937 | 0.980 | 0.926 | 0.954 | 0.667 |
| no MoE | 0.902 | 0.977 | 0.934 | 0.958 | 0.689 |
| MoE mag-only routing | 0.989 | 0.967 | 0.956 | 0.952 | 0.710 |
| no contrastive | 0.935 | 0.880 | 0.871 | 0.962 | 0.748 |
| no SupCon | 0.883 | 0.987 | 0.934 | 0.962 | 0.669 |
| no ordinal | 0.931 | 0.974 | 0.888 | 0.964 | 0.623 |

Decision threshold tuned on the val split (max val macro-F1), mean over folds:

| Variant | Acc@thr | PRR@thr | Macro-F1@thr | Mean val threshold (per-fold range) |
|---|---|---|---|---|
| full (rerun) | 0.891 | 0.898 | 0.871 | 0.73 (0.05–0.95) |
| no MoE | 0.892 | 0.902 | 0.867 | 0.57 (0.05–0.94) |
| MoE mag-only routing | 0.913 | 0.919 | 0.898 | 0.82 (0.61–0.95) |
| no contrastive | 0.903 | 0.913 | 0.886 | 0.54 (0.06–0.94) |
| no SupCon | 0.872 | 0.890 | 0.845 | 0.74 (0.05–0.95) |
| no ordinal | 0.871 | 0.889 | 0.846 | 0.76 (0.06–0.95) |

Fold 4 accuracy at 0.5 vs val-tuned threshold, full rerun: 0.667 → 0.693.

Trainable parameters: full / no-contrastive / no-SupCon / no-ordinal 48.54 M; mag-only routing 48.52 M;
no MoE 27.48 M.
