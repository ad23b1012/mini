import json, os, torch

sep = "=" * 60

# ── 1. Checkpoints ───────────────────────────────────────────
print(sep)
print("CHECKPOINTS")
print(sep)
ckpt_dir = "results/checkpoints"
for fname in sorted(os.listdir(ckpt_dir)):
    fpath = os.path.join(ckpt_dir, fname)
    size_mb = os.path.getsize(fpath) / 1e6
    try:
        ck = torch.load(fpath, map_location="cpu", weights_only=False)
        ep = ck.get("epoch", "?")
        mode = ck.get("model_mode", "?")
        m = ck.get("metrics", {})
        f1 = m.get("f1_weighted", "?")
        f1_str = f"{f1:.4f}" if isinstance(f1, float) else str(f1)
        print(f"  {fname:<40} {size_mb:>7.1f} MB  epoch={ep}  mode={mode}  f1={f1_str}")
    except Exception as e:
        print(f"  {fname:<40} {size_mb:>7.1f} MB  [load error: {e}]")

# ── 2. Evaluation results ────────────────────────────────────
print()
print(sep)
print("EVALUATION RESULTS")
print(sep)
for label, path in [
    ("Multimodal", "results/evaluation_multimodal/evaluation_results.json"),
    ("Vision-Only", "results/evaluation_vision_only/evaluation_results.json"),
]:
    if not os.path.exists(path):
        print(f"  {label}: NOT FOUND")
        continue
    with open(path) as f:
        r = json.load(f)
    print(f"\n  [{label}]")
    for k in ["accuracy", "f1_weighted", "f1_macro", "precision_weighted", "recall_weighted"]:
        v = r.get(k)
        if isinstance(v, float):
            print(f"    {k:<25}: {v:.4f}")
    print(f"    {'--- per class ---'}")
    for k, v in r.items():
        if k.startswith("f1_") and k != "f1_weighted" and k != "f1_macro":
            print(f"    {k:<25}: {v:.4f}")

# ── 3. Ambiguity analysis ────────────────────────────────────
print()
print(sep)
print("AMBIGUITY ANALYSIS")
print(sep)
amb_path = "results/ambiguity_analysis/ambiguity_summary.json"
if os.path.exists(amb_path):
    with open(amb_path) as f:
        a = json.load(f)
    total = a["total_samples"]
    mc = a["multimodal_only_correct"]
    bc = a["baseline_only_correct"]
    both_c = a["both_correct"]
    both_w = a["both_wrong"]
    dis = a["disagreements"]
    print(f"  Total test samples   : {total}")
    print(f"  Both correct         : {both_c} ({100*both_c/total:.1f}%)")
    print(f"  Multimodal-only wins : {mc} ({100*mc/total:.1f}%)")
    print(f"  Vision-only wins     : {bc} ({100*bc/total:.1f}%)")
    print(f"  Both wrong           : {both_w} ({100*both_w/total:.1f}%)")
    print(f"  Disagreements        : {dis} ({100*dis/total:.1f}%)")
    print(f"\n  Per-class multimodal improvement:")
    for cls, stats in a.get("per_class", {}).items():
        if stats["total"] > 0:
            imp = stats["multimodal_only_correct"]
            reg = stats["baseline_only_correct"]
            print(f"    {cls:<12}: +{imp} multimodal wins  -{reg} regressions  (n={stats['total']})")
    print(f"\n  Top confusion pairs (baseline wrong -> multimodal correct):")
    for pair, count in list(a.get("improvement_pairs", {}).items())[:5]:
        print(f"    {pair}: {count}")
else:
    print("  NOT FOUND")

# ── 4. XAI explanations ──────────────────────────────────────
print()
print(sep)
print("XAI EXPLANATIONS")
print(sep)
meta_path = "results/explanations_curated/run_metadata.json"
summ_path = "results/explanations_curated/curated_summary.json"
if os.path.exists(meta_path):
    with open(meta_path) as f:
        meta = json.load(f)
    print(f"  Generated at    : {meta.get('generated_at_utc','?')}")
    print(f"  Checkpoint      : {os.path.basename(meta.get('checkpoint','?'))}")
    print(f"  Total selected  : {meta.get('total_selected','?')}")
    print(f"  Total explained : {meta.get('total_explained','?')}")
    print(f"  Samples/class   : {meta.get('samples_per_class','?')}")
    print(f"  Min confidence  : {meta.get('min_confidence','?')}")
sample_dirs = [d for d in os.listdir("results/explanations_curated") if d.startswith("sample_")]
print(f"  Sample folders  : {len(sample_dirs)}")
complete = 0
for sd in sample_dirs:
    base = os.path.join("results/explanations_curated", sd)
    if all(os.path.exists(os.path.join(base, f)) for f in
           ["gradcam_overlay.png", "shap_tokens.png", "combined_explanation.png",
            "explanation_report.txt", "results.json"]):
        complete += 1
print(f"  Fully complete  : {complete}/{len(sample_dirs)}")

# ── 5. Training logs ─────────────────────────────────────────
print()
print(sep)
print("TRAINING LOGS")
print(sep)
log_dir = "results/logs"
if os.path.exists(log_dir):
    for f in os.listdir(log_dir):
        size_kb = os.path.getsize(os.path.join(log_dir, f)) / 1024
        print(f"  {f}  ({size_kb:.1f} KB)")
else:
    print("  NOT FOUND")

print()
print(sep)
print("DONE")
print(sep)
