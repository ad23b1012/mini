# MMER-XAI: Multimodal Emotion Recognition with Explainable AI

> **Explaining What the Face Shows and What the Words Mean**  
> A multimodal deep learning framework combining EfficientNet-B2 (face) + DeBERTa-v3-base (text) with Grad-CAM, SHAP, and a novel Cross-Modal Faithfulness Score for generating natural language explanations of emotion predictions.

---

## 📋 Abstract

We present MMER-XAI, a multimodal emotion recognition system that not only classifies emotions from facial expressions and textual context but also **explains its predictions in plain English**. Our framework combines visual (Grad-CAM) and textual (SHAP) explainability with a novel **Cross-Modal Faithfulness Score (CMFS)** that measures whether explanations from both modalities tell a consistent emotional story. We evaluate on AffectNet and MELD datasets, demonstrating that cross-attention fusion significantly improves both accuracy and explanation quality.

## 🔬 Key Contributions

1. **Multimodal Architecture**: EfficientNet-B2 + DeBERTa-v3-base with cross-attention fusion
2. **Cross-Modal Faithfulness Score (CMFS)**: Novel metric evaluating explanation consistency across visual and textual modalities
3. **NLG Explanation Pipeline**: Automatic generation of plain-English emotional analysis reports from XAI outputs
4. **Comprehensive Ablation**: Systematic comparison of fusion strategies and modality combinations

## 🏗️ Architecture

```
Input Face Image ──→ EfficientNet-B2 ──→ Visual Features ──→┐
                                                              │──→ Cross-Attention ──→ Classifier ──→ Emotion
Input Text       ──→ DeBERTa-v3-base ──→ Text Features  ──→┘       Fusion
                                                              
Visual Features ──→ Grad-CAM ──→ Region Scores ──→┐
                                                    │──→ NLG Report ──→ "The model detected sadness..."
Text Features   ──→ SHAP     ──→ Token Scores  ──→┘
                                                    │──→ CMFS Score ──→ 0.89 (strong agreement)
```

## 📁 Project Structure

```
├── config/config.yaml          # All hyperparameters
├── data/                       # Dataset loaders (AffectNet, MELD)
├── models/                     # Vision + Text encoders + Fusion
├── explainers/                 # Grad-CAM, SHAP, Faithfulness, NLG
├── training/                   # Trainer, losses, metrics
├── scripts/                    # train.py, evaluate.py, explain.py, ablation.py
└── utils/                      # Logging, visualization, helpers
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync all dependencies (creates venv + installs everything)
uv sync

# For CUDA support, install PyTorch with CUDA index
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

> **Note:** `uv sync` reads `pyproject.toml` and creates a reproducible `.venv` automatically. No conda needed.

### 2. Download Datasets

```bash
# Download MELD (public dataset)
python data/download_datasets.py --dataset meld

# For AffectNet: register at http://mohammadmahoor.com/affectnet/
python data/download_datasets.py --dataset affectnet --verify
```

### 3. Train

```bash
# Train multimodal model on MELD
python scripts/train.py --config config/config.yaml --dataset meld

# Quick debug run (1 epoch)
python scripts/train.py --config config/config.yaml --debug

# Vision-only (no text) on AffectNet
python scripts/train.py --config config/config.yaml --dataset affectnet --mode vision_only
```

### 4. Evaluate

```bash
python scripts/evaluate.py \
    --config config/config.yaml \
    --checkpoint results/checkpoints/best_model.pt \
    --dataset meld --split test
```

### 5. Generate Explanations

```bash
# Explain 10 random samples
python scripts/explain.py \
    --checkpoint results/checkpoints/best_model.pt \
    --num-samples 10

# Explain a specific sample
python scripts/explain.py \
    --checkpoint results/checkpoints/best_model.pt \
    --sample-idx 42
```

### 6. Ablation Study

```bash
# Full ablation (all 5 model variants)
python scripts/ablation.py --config config/config.yaml

# Quick ablation (10 epochs per variant)
python scripts/ablation.py --config config/config.yaml --quick
```

## 📊 Expected Results

| Model | Accuracy | F1 (Weighted) | F1 (Macro) |
|-------|----------|---------------|------------|
| Vision-Only (EfficientNet-B2) | ~58% | ~55% | ~42% |
| Text-Only (DeBERTa-v3-base) | ~63% | ~62% | ~48% |
| Multimodal + Concat | ~65% | ~64% | ~51% |
| Multimodal + Gated | ~66% | ~65% | ~53% |
| **Multimodal + Cross-Attention (Ours)** | **~68%** | **~67%** | **~55%** |

*Results on MELD test set. Exact numbers depend on training runs.*

## 📝 Example Explanation Output

```
============================================================
MULTIMODAL EMOTION ANALYSIS REPORT
============================================================

Predicted Emotion: **Sadness**
Confidence: 87.3%
Speaker: Rachel
Utterance: "I can't do this anymore, I just feel so alone"

This prediction indicates feelings of loss, grief, or disappointment.

--- Visual Cues (Face Analysis) ---

The model focused primarily on the mouth region (importance: 0.42)
showing a downturned expression. Secondary attention areas: left eye
region (0.31), jaw and chin area (0.15).

--- Textual Cues (Language Analysis) ---

The word/phrase "alone" contributed most strongly toward sadness
(SHAP: +0.651). Other supporting words: "can't" (+0.423), "anymore"
(+0.312).

--- Explanation Quality Metrics ---

Cross-Modal Faithfulness Score (CMFS): 0.847
Cross-Modal Agreement: 0.892 (strong agreement) — visual and textual
explanations consistently support the sadness prediction.
============================================================
```

## 🔧 Configuration

All hyperparameters are in `config/config.yaml`. Key settings:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model.vision.backbone` | `efficientnet_b2` | Vision encoder |
| `model.text.backbone` | `microsoft/deberta-v3-base` | Text encoder |
| `model.fusion.strategy` | `cross_attention` | Fusion method |
| `training.batch_size` | `32` | Batch size (adjust for GPU VRAM) |
| `training.epochs` | `50` | Training epochs |
| `training.amp` | `true` | Mixed precision training |

## 📄 Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{mmer-xai-2026,
  title={MMER-XAI: Multimodal Emotion Recognition with Cross-Modal
         Explainable AI and Natural Language Generation},
  author={Abhi},
  year={2026},
}
```

## 📜 License

MIT License
