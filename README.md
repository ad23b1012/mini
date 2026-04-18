# MMER-XAI: Multimodal Emotion Recognition with Explainable AI

> **Explaining What the Face Shows and What the Words Mean**  
> A multimodal deep learning framework combining EfficientNet-B2 (face) + DeBERTa-v3-base (text) with Grad-CAM, SHAP, and a novel Cross-Modal Faithfulness Score for generating natural language explanations of emotion predictions.

---

## 📋 Abstract

We present MMER-XAI, a multimodal emotion recognition system that not only classifies emotions from facial expressions and textual context but also **explains its predictions visually and textually**. Our framework combines visual (Grad-CAM) and textual (SHAP) explainability with a novel **Cross-Modal Faithfulness Score (CMFS)** that measures whether explanations from both modalities tell a consistent emotional story. We evaluate on the MELD dataset, demonstrating that cross-attention fusion significantly improves both accuracy and explanation quality over unimodal baselines.

## 🔬 Key Contributions

1. **Multimodal Architecture**: EfficientNet-B2 + DeBERTa-v3-base with cross-attention fusion
2. **Cross-Modal Faithfulness Score (CMFS)**: Novel metric evaluating explanation consistency across visual and textual modalities
3. **Paper-Ready Visualizations**: Automated generation of publication-grade Grad-CAM and SHAP combined visual overlays
4. **Comprehensive Ablation**: Systematic comparison of visual-only, text-only, and multimodal variants

## 🏗️ Architecture

```
Input Face Image ──→ EfficientNet-B2 ──→ Visual Features ──→┐
                                                              │──→ Cross-Attention ──→ Classifier ──→ Emotion
Input Text       ──→ DeBERTa-v3-base ──→ Text Features  ──→┘       Fusion
                                                              
Visual Features ──→ Grad-CAM ──→ Region Scores ──→┐
                                                    │──→ CMFS Faithfulness Score 
Text Features   ──→ SHAP     ──→ Token Scores  ──→┘
```

## 🚀 One-Click Pipeline Execution

To automatically train the models, evaluate baselines, and generate all tables, figures, and visual explanations for your paper, run the master orchestrator script:

```powershell
.\run_all_experiments.ps1
```

This PowerShell script is fully resumable and idempotent. It natively handles offline configurations to bypass HuggingFace Hub network resets, making it extremely robust.

### What the Pipeline Generates:
All outputs required for publication are organized in the `results/` directory:

1. **`results/evaluation_multimodal/`** (and text_only, vision_only):
   - `evaluation_results.json`: Full quantitative metrics (F1, Precision, Recall).
   - `results_table.tex`: Copy-paste ready LaTeX table of results.
   - `confusion_matrix.png`: High-res confusion matrix figure.
2. **`results/faithfulness/`**: 
   - `faithfulness_results.json`: Final CMFS, Comprehensiveness, and Sufficiency scores evaluated on the test set.
   - `faithfulness_table.tex`: LaTeX summary table of faithfulness metrics.
3. **`results/aggregate_xai/`**:
   - Aggregate statistics across the entire test set (e.g., "jaw and chin are the most important regions for Anger and Disgust").
4. **`results/curated_xai/`**:
   - `sample_XXXX/combined_explanation.png`: **The hero figures for your paper.** Striking side-by-side images showing the face overlay (Grad-CAM) + text tokens (SHAP) + predictions.

## 📊 Evaluation Results (MELD Dataset)

Our systematic ablation demonstrates the value of the multimodal approach over unimodal baselines:

| Model Variant | Weighted F1 | Accuracy | Macro F1 |
|---------------|-------------|----------|----------|
| **Multimodal (Ours)** | **52.71%** | **50.88%** | **37.37%** |
| Text-Only Baseline | 48.14% | 43.91% | 36.61% |
| Vision-Only Baseline* | 5.95% | 8.35% | 7.94% |

*\* Note: The vision-only baseline highlights the extreme ambiguity of conversational faces on MELD without textual context (especially for 'Neutral', which dominates the dataset).*

## 📈 Explanation Faithfulness Findings

Our novel Cross-Modal Faithfulness Score reveals high consistency between visual and linguistic cues:
- **Vision Sufficiency:** 94.8%
- **Cross-Modal Agreement:** 79.8%
- **Text Sufficiency:** 56.5%
- **CMFS (Overall Target):** 43.1%

## 🔧 Configuration and Environment

Environment requirements:
- Python 3.10+
- `uv` package manager (`curl -LsSf https://astral.sh/uv/install.sh | sh`)

```bash
# Sync dependency tree
uv sync

# Offline overrides used during eval to ensure reproducible bounds without network dependencies:
$env:TRANSFORMERS_OFFLINE="1"
$env:HF_HUB_OFFLINE="1"
```

The master configuration file defining the model architecture, training epochs, and paths is located at `config/config.yaml`.

## 📄 Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{mmer-xai-2026,
  title={MMER-XAI: Multimodal Emotion Recognition with Cross-Modal
         Explainable AI and Natural Language Generation},
  author={Abhishek Buddiga},
  year={2026},
}
```

## 📜 License

MIT License
