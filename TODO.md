# To do

- [x] Exp 06 ZoomMoE + VLM backbone: done 2026-10-22 (results in [RESULTS.md](RESULTS.md)).
- [x] Exp 07 binary ablation: done 2026-10-22.
- [ ] **Exp 07 for 8-class (most important next run).** Binary is near ceiling, and the binary ablation found
      mag-only routing better than the full model, so the 8-class ablation decides which ZoomMoE design to
      report. Free accounts: Gmail nabinkoirala467 (~28 h), pulchowk (20.9 h). Needs `TASK`/paths switched
      to the `-mc` datasets and the 8-class fold checks (from mc notebook 02's printed audit).
- [ ] Decide the headline model per task: 8-class ZoomMoE + QuiltNet, binary ZoomMoE + ResNet-50
      (+ maybe mag-only routing, after the 8-class ablation).
- [x] Exp 08 (notebook-03 tests, all folds) and exp 09 (same-backbone baselines): done 2026-10-22.
- [ ] **Decide the paper's story with the guide**: simple fusion ≥ ZoomMoE (FINDINGS 14–16). Options: improve ZoomMoE
      (supervised stage 0, mag-only routing, drop PZR ordering) or reframe as an honest benchmark of
      multi-scale fusion + VLMs under a patient-disjoint protocol.
- [ ] Statistics: patient-level bootstrap CIs and paired tests, ZoomMoE vs each VLM (no GPU).
- [ ] Optional: LoRA fine-tuning of a generative VLM (Qwen2-VL-2B) to complete the VLM ladder.
- [ ] Consolidate everything into the IEEE TMI 8-page draft.

Background: training-time ablations had never been run before exp 07 (`RUN_ABLATIONS = False` in both
notebook 02s; the built-in driver was fold 0 only, with a short 2/4/2-epoch schedule).
