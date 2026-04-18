import ast, sys

# 1. Syntax check all pipeline files
files = [
    'training/trainer.py', 'training/losses.py', 'training/metrics.py',
    'scripts/train.py', 'scripts/evaluate.py', 'scripts/analyze_ambiguity.py',
    'scripts/explain.py', 'scripts/explain_curated.py',
    'data/meld_dataset.py', 'data/affectnet_dataset.py',
    'models/multimodal_model.py', 'models/text_encoder.py',
    'models/vision_encoder.py', 'models/fusion.py',
    'explainers/gradcam.py', 'explainers/shap_text.py',
    'explainers/faithfulness.py',
    'utils/helpers.py', 'utils/visualization.py', 'utils/face_quality.py',
]
print('=== SYNTAX CHECK ===')
all_ok = True
for f in files:
    try:
        ast.parse(open(f, encoding='utf-8', errors='ignore').read())
        print(f'  OK  {f}')
    except SyntaxError as e:
        print(f'  ERR {f} line {e.lineno}: {e.msg}')
        all_ok = False
    except FileNotFoundError:
        print(f'  SKIP {f} (not found)')

# 2. Verify checkpoint metadata
print()
print('=== CHECKPOINT METADATA IN _save_checkpoint ===')
src = open('training/trainer.py', encoding='utf-8', errors='ignore').read()
required_fields = ['model_mode', 'fusion_strategy', 'class_names', 'num_classes', 'dataset_name']
for field in required_fields:
    found = f'"{field}":' in src or f"'{field}':" in src
    print(f'  {"FOUND" if found else "MISSING"}  {field}')

# 3. Warmup guard
print()
print('=== WARMUP GUARD ===')
print(f'  Warmup guard in trainer.py: {"Warmup guard" in src}')

# 4. Offline tokenizer
print()
print('=== TOKENIZER ===')
meld_src = open('data/meld_dataset.py', encoding='utf-8', errors='ignore').read()
print(f'  Offline-first tokenizer: {"local_files_only=True" in meld_src}')

# 5. Metadata stamps in train.py
print()
print('=== TRAIN.PY METADATA STAMPS ===')
train_src = open('scripts/train.py', encoding='utf-8', errors='ignore').read()
for attr in ['trainer.model_mode', 'trainer.fusion_strategy', 'trainer.train_class_distribution']:
    print(f'  {"FOUND" if attr in train_src else "MISSING"}  {attr}')

# 6. Vision-only overrides
print()
print('=== VISION-ONLY OVERRIDES ===')
print(f'  freeze_layers mutated: {"freeze_layers\"] = vision_freeze_layers" in train_src}')
print(f'  patience=15: {"es_patience = 15" in train_src}')

# 7. Evaluate reads metadata
print()
print('=== EVALUATE.PY ===')
eval_src = open('scripts/evaluate.py', encoding='utf-8', errors='ignore').read()
print(f'  model_mode lookup: {"model_mode" in eval_src}')
print(f'  fusion_strategy lookup: {"fusion_strategy" in eval_src}')
print(f'  _build_model_from_checkpoint: {"_build_model_from_checkpoint" in eval_src}')

print()
print('=== ALL CHECKS COMPLETE ===' if all_ok else '=== ERRORS FOUND ===')
