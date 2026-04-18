"""
Ambiguity-resolution analysis for face-only vs multimodal checkpoints.

This script compares two trained checkpoints on the same evaluation split and
extracts cases where the multimodal model is correct while the baseline model
is wrong. These are the key examples for the thesis claim that context text
resolves facial ambiguity.

Usage:
    python scripts/analyze_ambiguity.py ^
        --config config/config.yaml ^
        --baseline-checkpoint path/to/vision_only.pt ^
        --multimodal-checkpoint path/to/multimodal.pt
"""

import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.affectnet_dataset import AffectNetDataset
from data.meld_dataset import MELDDataset
from scripts.evaluate import (
    _build_meld_dataset_kwargs,
    _build_model_from_checkpoint,
    _resolve_checkpoint_fusion,
    _resolve_checkpoint_mode,
)
from utils.helpers import get_device, load_config, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze ambiguity-resolving cases")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--baseline-checkpoint", type=str, required=True)
    parser.add_argument("--multimodal-checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, choices=["meld", "affectnet"])
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--min-confidence-gap", type=float, default=0.0)
    parser.add_argument("--focus-classes", nargs="*", default=None)
    parser.add_argument("--baseline-label", type=str, default="Face-Only")
    parser.add_argument("--multimodal-label", type=str, default="Multimodal")
    parser.add_argument("--output-dir", type=str, default="./results/ambiguity_analysis")
    return parser.parse_args()


def _normalize_focus_classes(focus_classes):
    if not focus_classes:
        return None
    return {name.strip().lower() for name in focus_classes}


def _build_dataset(config: dict, dataset_name: str, split: str):
    """Build the shared evaluation dataset used for both checkpoints."""
    dataset_cfg = config["dataset"].get(dataset_name, {})
    if dataset_name == "meld":
        return MELDDataset(**_build_meld_dataset_kwargs(config, split=split))

    split_map = {"test": "val", "val": "val", "train": "train"}
    return AffectNetDataset(
        root_dir=dataset_cfg.get("root_dir", "./datasets/affectnet"),
        split=split_map.get(split, "val"),
        num_classes=dataset_cfg.get("num_classes", 8),
        image_size=dataset_cfg.get("image_size", 260),
    )


def _to_python_list(value):
    """Convert tensors/arrays/scalars into ordinary Python values."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, tuple):
        return list(value)
    return value


def _build_forward_kwargs(batch: dict, mode: str, device: str) -> dict:
    """Construct the correct model inputs for a given mode."""
    kwargs = {}
    if mode in ("multimodal", "vision_only") and "image" in batch:
        kwargs["image"] = batch["image"].to(device)
    if mode in ("multimodal", "text_only") and "input_ids" in batch:
        kwargs["input_ids"] = batch["input_ids"].to(device)
        kwargs["attention_mask"] = batch["attention_mask"].to(device)
    return kwargs


def _safe_batch_list(batch: dict, key: str, batch_size: int, default):
    value = batch.get(key)
    if value is None:
        return [default for _ in range(batch_size)]
    value = _to_python_list(value)
    if isinstance(value, list):
        return value
    return [value for _ in range(batch_size)]


def _write_markdown_report(report_path: str, summary: dict, top_cases: list):
    """Write a concise markdown report for thesis notes."""
    lines = [
        "# Ambiguity Resolution Analysis",
        "",
        f"- Baseline: `{summary['baseline_label']}`",
        f"- Multimodal: `{summary['multimodal_label']}`",
        f"- Split: `{summary['split']}`",
        f"- Total samples: `{summary['total_samples']}`",
        f"- Multimodal-only correct: `{summary['multimodal_only_correct']}`",
        f"- Baseline-only correct: `{summary['baseline_only_correct']}`",
        f"- Both correct: `{summary['both_correct']}`",
        f"- Both wrong: `{summary['both_wrong']}`",
        "",
        "## Per-Class Improvements",
        "",
    ]

    for class_name, stats in summary["per_class"].items():
        lines.append(
            f"- {class_name}: total={stats['total']}, "
            f"multimodal_only_correct={stats['multimodal_only_correct']}, "
            f"baseline_only_correct={stats['baseline_only_correct']}"
        )

    lines.extend(["", "## Top Ambiguity Cases", ""])
    if not top_cases:
        lines.append("- No multimodal-only improvements matched the current filters.")
    else:
        for case in top_cases[:10]:
            lines.append(
                f"- dia{case.get('dialogue_id', '?')}/utt{case.get('utterance_id', '?')}: "
                f"true={case['true_emotion']}, "
                f"{summary['baseline_label']}={case['baseline_prediction']} "
                f"({case['baseline_confidence']:.3f}), "
                f"{summary['multimodal_label']}={case['multimodal_prediction']} "
                f"({case['multimodal_confidence']:.3f})"
            )
            if case.get("text_input"):
                lines.append(f"  text: {case['text_input']}")

    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main():
    args = parse_args()
    config = load_config(args.config)
    device = get_device()
    set_seed(config.get("project", {}).get("seed", 42))

    dataset_name = args.dataset or config["dataset"]["name"]
    focus_classes = _normalize_focus_classes(args.focus_classes)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\nLoading shared {dataset_name} {args.split} set...")
    dataset = _build_dataset(config, dataset_name, args.split)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=config.get("training", {}).get("num_workers", 0),
        pin_memory=config.get("training", {}).get("pin_memory", False),
    )

    print("\nLoading checkpoints...")
    baseline_checkpoint = torch.load(
        args.baseline_checkpoint, map_location=device, weights_only=False
    )
    multimodal_checkpoint = torch.load(
        args.multimodal_checkpoint, map_location=device, weights_only=False
    )

    baseline_model = _build_model_from_checkpoint(config, baseline_checkpoint).to(device)
    baseline_model.load_state_dict(baseline_checkpoint["model_state_dict"])
    baseline_model.eval()

    multimodal_model = _build_model_from_checkpoint(config, multimodal_checkpoint).to(device)
    multimodal_model.load_state_dict(multimodal_checkpoint["model_state_dict"])
    multimodal_model.eval()

    baseline_mode = _resolve_checkpoint_mode(baseline_checkpoint, config)
    multimodal_mode = _resolve_checkpoint_mode(multimodal_checkpoint, config)

    print(f"  Baseline mode: {baseline_mode}")
    print(f"  Baseline fusion: {_resolve_checkpoint_fusion(baseline_checkpoint, config)}")
    print(f"  Multimodal mode: {multimodal_mode}")
    print(f"  Multimodal fusion: {_resolve_checkpoint_fusion(multimodal_checkpoint, config)}")

    class_names = (
        multimodal_checkpoint.get("class_names")
        or baseline_checkpoint.get("class_names")
        or config["dataset"][dataset_name].get("class_names")
    )

    summary = {
        "dataset": dataset_name,
        "split": args.split,
        "baseline_label": args.baseline_label,
        "multimodal_label": args.multimodal_label,
        "baseline_mode": baseline_mode,
        "multimodal_mode": multimodal_mode,
        "baseline_fusion": _resolve_checkpoint_fusion(baseline_checkpoint, config),
        "multimodal_fusion": _resolve_checkpoint_fusion(multimodal_checkpoint, config),
        "total_samples": 0,
        "both_correct": 0,
        "both_wrong": 0,
        "multimodal_only_correct": 0,
        "baseline_only_correct": 0,
        "disagreements": 0,
        "per_class": {},
        "improvement_pairs": {},
    }
    per_class = defaultdict(
        lambda: {
            "total": 0,
            "both_correct": 0,
            "both_wrong": 0,
            "multimodal_only_correct": 0,
            "baseline_only_correct": 0,
        }
    )
    improvement_pairs = defaultdict(int)
    improvement_cases = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Comparing checkpoints"):
            labels = batch["label"].cpu()
            batch_size = labels.shape[0]

            baseline_output = baseline_model(
                **_build_forward_kwargs(batch, baseline_mode, device)
            )
            multimodal_output = multimodal_model(
                **_build_forward_kwargs(batch, multimodal_mode, device)
            )

            baseline_probs = torch.softmax(baseline_output["logits"], dim=1).cpu()
            multimodal_probs = torch.softmax(multimodal_output["logits"], dim=1).cpu()

            baseline_conf, baseline_preds = baseline_probs.max(dim=1)
            multimodal_conf, multimodal_preds = multimodal_probs.max(dim=1)

            utterances = _safe_batch_list(batch, "utterance", batch_size, "")
            text_inputs = _safe_batch_list(batch, "text_input", batch_size, "")
            speakers = _safe_batch_list(batch, "speaker", batch_size, "")
            dialogue_ids = _safe_batch_list(batch, "dialogue_id", batch_size, None)
            utterance_ids = _safe_batch_list(batch, "utterance_id", batch_size, None)
            face_quality_scores = _safe_batch_list(
                batch, "face_quality_score", batch_size, 0.0
            )

            for idx in range(batch_size):
                label_idx = int(labels[idx].item())
                baseline_pred = int(baseline_preds[idx].item())
                multimodal_pred = int(multimodal_preds[idx].item())
                true_emotion = class_names[label_idx]
                baseline_emotion = class_names[baseline_pred]
                multimodal_emotion = class_names[multimodal_pred]

                if focus_classes and true_emotion.lower() not in focus_classes:
                    continue

                summary["total_samples"] += 1
                per_class[true_emotion]["total"] += 1

                baseline_correct = baseline_pred == label_idx
                multimodal_correct = multimodal_pred == label_idx

                if baseline_pred != multimodal_pred:
                    summary["disagreements"] += 1

                if baseline_correct and multimodal_correct:
                    summary["both_correct"] += 1
                    per_class[true_emotion]["both_correct"] += 1
                elif baseline_correct and not multimodal_correct:
                    summary["baseline_only_correct"] += 1
                    per_class[true_emotion]["baseline_only_correct"] += 1
                elif not baseline_correct and multimodal_correct:
                    summary["multimodal_only_correct"] += 1
                    per_class[true_emotion]["multimodal_only_correct"] += 1

                    confidence_gap = float(
                        multimodal_conf[idx].item() - baseline_conf[idx].item()
                    )
                    improvement_pairs[f"{baseline_emotion} -> {true_emotion}"] += 1

                    if confidence_gap >= args.min_confidence_gap:
                        improvement_cases.append(
                            {
                                "dialogue_id": dialogue_ids[idx],
                                "utterance_id": utterance_ids[idx],
                                "speaker": speakers[idx],
                                "utterance": utterances[idx],
                                "text_input": text_inputs[idx],
                                "true_label": label_idx,
                                "true_emotion": true_emotion,
                                "baseline_prediction": baseline_emotion,
                                "baseline_confidence": float(baseline_conf[idx].item()),
                                "multimodal_prediction": multimodal_emotion,
                                "multimodal_confidence": float(multimodal_conf[idx].item()),
                                "confidence_gap": confidence_gap,
                                "face_quality_score": float(face_quality_scores[idx]),
                            }
                        )
                else:
                    summary["both_wrong"] += 1
                    per_class[true_emotion]["both_wrong"] += 1

    sorted_cases = sorted(
        improvement_cases,
        key=lambda item: (
            item["confidence_gap"],
            item["multimodal_confidence"],
            -item["baseline_confidence"],
        ),
        reverse=True,
    )
    top_cases = sorted_cases[: args.top_k]

    summary["per_class"] = dict(sorted(per_class.items()))
    summary["improvement_pairs"] = dict(
        sorted(improvement_pairs.items(), key=lambda item: item[1], reverse=True)
    )
    summary["num_saved_improvement_cases"] = len(sorted_cases)
    summary["top_k"] = args.top_k
    summary["min_confidence_gap"] = args.min_confidence_gap

    summary_path = os.path.join(args.output_dir, "ambiguity_summary.json")
    cases_path = os.path.join(args.output_dir, "ambiguity_cases.json")
    top_cases_path = os.path.join(args.output_dir, "ambiguity_cases_topk.json")
    report_path = os.path.join(args.output_dir, "ambiguity_report.md")

    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    with open(cases_path, "w", encoding="utf-8") as handle:
        json.dump(sorted_cases, handle, indent=2)
    with open(top_cases_path, "w", encoding="utf-8") as handle:
        json.dump(top_cases, handle, indent=2)
    _write_markdown_report(report_path, summary, top_cases)

    print("\n" + "=" * 60)
    print("AMBIGUITY ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"  Total samples analyzed: {summary['total_samples']}")
    print(f"  Multimodal-only correct: {summary['multimodal_only_correct']}")
    print(f"  Baseline-only correct: {summary['baseline_only_correct']}")
    print(f"  Both correct: {summary['both_correct']}")
    print(f"  Both wrong: {summary['both_wrong']}")
    print(f"  Saved improvement cases: {summary['num_saved_improvement_cases']}")
    print(f"  Summary saved to: {summary_path}")
    print(f"  Report saved to: {report_path}")


if __name__ == "__main__":
    main()
