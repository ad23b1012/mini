# Research Runbook

This runbook is the shortest path from code to thesis-ready outputs.

## 1. Audit Before Training

```powershell
python scripts/diagnose.py --config config/config.yaml --sample-batches 3
```

What this verifies:
- MELD face-quality filtering is active
- class-balanced sampling is active
- `focal_alpha` is auto-resolved from the filtered train split
- transfer loading works for a given checkpoint

Optional transfer check:

```powershell
python scripts/diagnose.py `
  --config config/config.yaml `
  --transfer-checkpoint results/checkpoints/best_model.pt `
  --transfer-component vision_encoder
```

## 2. Vision Pretraining

Pretrain the face encoder on AffectNet:

```powershell
python scripts/train.py `
  --config config/config.yaml `
  --dataset affectnet `
  --mode vision_only
```

Expected result:
- a vision-only checkpoint in `results/checkpoints/`

## 3. Multimodal Fine-Tuning on MELD

Fine-tune on MELD with transferred vision weights:

```powershell
python scripts/train.py `
  --config config/config.yaml `
  --dataset meld `
  --mode multimodal `
  --init-checkpoint <AFFECTNET_CHECKPOINT> `
  --transfer-component vision_encoder
```

Notes:
- training uses balanced sampling automatically
- training filters low-quality face crops automatically
- dialogue-history support exists, but keep it disabled until the baseline is stable

## 4. Evaluate Final Checkpoints

Evaluate the main multimodal checkpoint:

```powershell
python scripts/evaluate.py `
  --config config/config.yaml `
  --checkpoint <MULTIMODAL_CHECKPOINT> `
  --dataset meld `
  --split test `
  --output-dir results/evaluation_multimodal
```

Evaluate a face-only checkpoint:

```powershell
python scripts/evaluate.py `
  --config config/config.yaml `
  --checkpoint <VISION_ONLY_CHECKPOINT> `
  --dataset meld `
  --split test `
  --output-dir results/evaluation_face_only
```

## 5. Run Ablations

```powershell
python scripts/ablation.py --config config/config.yaml
```

Quick sanity run:

```powershell
python scripts/ablation.py --config config/config.yaml --quick
```

This produces the comparison needed for:
- face-only vs text-only vs multimodal
- fusion strategy comparison

## 6. Ambiguity-Resolution Analysis

This is the key script for the thesis claim that context resolves ambiguity.

```powershell
python scripts/analyze_ambiguity.py `
  --config config/config.yaml `
  --baseline-checkpoint <VISION_ONLY_CHECKPOINT> `
  --multimodal-checkpoint <MULTIMODAL_CHECKPOINT> `
  --dataset meld `
  --split test `
  --output-dir results/ambiguity_analysis
```

Outputs:
- `ambiguity_summary.json`
- `ambiguity_cases.json`
- `ambiguity_cases_topk.json`
- `ambiguity_report.md`

Use these to write:
- where face-only failed
- where multimodal succeeded
- which text context likely resolved the ambiguity

## 7. Generate Explanations

Random explanation set:

```powershell
python scripts/explain.py
```

Curated explanation set:

```powershell
python scripts/explain_curated.py
```

## 8. Suggested Reporting Order

1. AffectNet vision-only pretraining result
2. MELD face-only result
3. MELD multimodal result
4. Ablation table
5. Ambiguity-resolution analysis
6. XAI examples
7. NLG report examples

## 9. What To Compare In The Thesis

- weighted F1 and macro F1
- per-class F1, especially `Fear` and `Disgust`
- multimodal-only correct ambiguity cases
- curated explanations for correct predictions
- quality-filtered vs unfiltered training discussion
