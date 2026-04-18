$ErrorActionPreference = "Stop"

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host " MMER-XAI: Full Pipeline Execution Script (RESUMABLE)" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# 1. Save Multimodal Baseline
Write-Host "`n[1/5] Checking Multimodal Model Checkpoint..." -ForegroundColor Yellow
if (-not (Test-Path "results\checkpoints\meld_multimodal_best.pt")) {
    if (Test-Path "results\checkpoints\best_model.pt") {
        Copy-Item "results\checkpoints\best_model.pt" "results\checkpoints\meld_multimodal_best.pt" -Force
        Write-Host "  -> Saved successfully as meld_multimodal_best.pt" -ForegroundColor Green
    } else {
        Write-Host "  -> WARNING: results\checkpoints\best_model.pt not found. Ensure multimodal training completed properly." -ForegroundColor Red
    }
} else {
    Write-Host "  -> Multimodal model already saved. Skipping." -ForegroundColor DarkGray
}

# 2. Train Vision-Only
Write-Host "`n[2/5] Checking Vision-Only Baseline..." -ForegroundColor Yellow
if (-not (Test-Path "results\checkpoints\meld_vision_only_best.pt")) {
    Write-Host "  -> Training Vision-Only..."
    # Always starts fresh if it was halted previously without saving the best, 
    # but since yours finished already, this will be skipped!
    uv run python scripts/train.py --config config/config.yaml --dataset meld --mode vision_only
    Copy-Item "results\checkpoints\best_model.pt" "results\checkpoints\meld_vision_only_best.pt" -Force
    Write-Host "  -> Vision-Only training complete." -ForegroundColor Green
} else {
    Write-Host "  -> Vision-Only model already trained. Skipping." -ForegroundColor DarkGray
}

# 3. Train Text-Only
Write-Host "`n[3/5] Checking Text-Only Baseline..." -ForegroundColor Yellow
if (-not (Test-Path "results\checkpoints\meld_text_only_best.pt")) {
    Write-Host "  -> Continuing Text-Only Training..."
    
    # Check if there is an ongoing text-only run to resume
    if (Test-Path "results\checkpoints\best_model.pt") {
        Write-Host "  -> Found existing checkpoint. Resuming from where it stopped!" -ForegroundColor Green
        uv run python scripts/train.py --config config/config.yaml --dataset meld --mode text_only --resume results\checkpoints\best_model.pt
    } else {
        uv run python scripts/train.py --config config/config.yaml --dataset meld --mode text_only
    }
    Copy-Item "results\checkpoints\best_model.pt" "results\checkpoints\meld_text_only_best.pt" -Force
    Write-Host "  -> Text-Only training complete." -ForegroundColor Green
} else {
    Write-Host "  -> Text-Only model already trained. Skipping." -ForegroundColor DarkGray
}

# 4. Evaluations
Write-Host "`n[4/5] Running Evaluations on Test Set..." -ForegroundColor Yellow

if (-not (Test-Path "results\evaluation_multimodal\metrics.json")) {
    Write-Host "  -> Evaluating Multimodal..."
    uv run python scripts/evaluate.py --config config/config.yaml --checkpoint results\checkpoints\meld_multimodal_best.pt --dataset meld --split test --output-dir results\evaluation_multimodal
} else {
    Write-Host "  -> Multimodal evaluation already exists. Skipping." -ForegroundColor DarkGray
}

if (-not (Test-Path "results\evaluation_vision_only\metrics.json")) {
    Write-Host "  -> Evaluating Vision-Only..."
    uv run python scripts/evaluate.py --config config/config.yaml --checkpoint results\checkpoints\meld_vision_only_best.pt --dataset meld --split test --output-dir results\evaluation_vision_only
} else {
    Write-Host "  -> Vision-Only evaluation already exists. Skipping." -ForegroundColor DarkGray
}

if (-not (Test-Path "results\evaluation_text_only\metrics.json")) {
    Write-Host "  -> Evaluating Text-Only..."
    uv run python scripts/evaluate.py --config config/config.yaml --checkpoint results\checkpoints\meld_text_only_best.pt --dataset meld --split test --output-dir results\evaluation_text_only
} else {
    Write-Host "  -> Text-Only evaluation already exists. Skipping." -ForegroundColor DarkGray
}

# 5. XAI and Analysis
Write-Host "`n[5/5] Generating XAI Statistics and Ambiguity Analysis (takes ~30 mins)..." -ForegroundColor Yellow

if (-not (Test-Path "results\ambiguity_analysis\ambiguity_report.md")) {
    Write-Host "  -> Running Ambiguity Analysis..."
    uv run python scripts/analyze_ambiguity.py --config config/config.yaml --baseline-checkpoint results\checkpoints\meld_vision_only_best.pt --multimodal-checkpoint results\checkpoints\meld_multimodal_best.pt --dataset meld --split test --output-dir results\ambiguity_analysis
} else {
    Write-Host "  -> Ambiguity Analysis already done. Skipping." -ForegroundColor DarkGray
}

if (-not (Test-Path "results\aggregate_xai\aggregate_xai_table.tex")) {
    Write-Host "  -> Computing Aggregate XAI (Grad-CAM & SHAP)..."
    uv run python scripts/aggregate_xai.py --config config/config.yaml --checkpoint results\checkpoints\meld_multimodal_best.pt --output-dir results\aggregate_xai
} else {
    Write-Host "  -> Aggregate XAI already computed. Skipping." -ForegroundColor DarkGray
}

if (-not (Test-Path "results\faithfulness\faithfulness_table.tex")) {
    Write-Host "  -> Computing Faithfulness Metrics..."
    uv run python scripts/run_faithfulness.py --config config/config.yaml --checkpoint results\checkpoints\meld_multimodal_best.pt --output-dir results\faithfulness
} else {
    Write-Host "  -> Faithfulness metrics already computed. Skipping." -ForegroundColor DarkGray
}

Write-Host "`n==========================================================" -ForegroundColor Cyan
Write-Host " ALL EXPERIMENTS COMPLETED SUCCESSFULLY" -ForegroundColor Green
Write-Host " LaTeX tables are ready in results/aggregate_xai and results/faithfulness" -ForegroundColor Cyan
Write-Host " Evaluated Accuracies are stored inside results/evaluation_* folders" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan
