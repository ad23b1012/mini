import torch

for name, path in [
    ("MULTIMODAL", "results/checkpoints/meld_multimodal_best.pt"),
    ("VISION-ONLY", "results/checkpoints/meld_vision_only_best.pt"),
]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = ck.get("metrics", {})
    print(f"=== {name} CHECKPOINT ===")
    print(f"  epoch          : {ck.get('epoch')}")
    print(f"  model_mode     : {ck.get('model_mode', 'N/A')}")
    print(f"  fusion_strategy: {ck.get('fusion_strategy', 'N/A')}")
    print(f"  val f1_weighted: {m.get('f1_weighted', 'N/A')}")
    print(f"  val accuracy   : {m.get('accuracy', 'N/A')}")
    print(f"  class_names    : {ck.get('class_names', [])}")
    dist = ck.get("train_class_distribution")
    if dist:
        print(f"  train_dist     : {dist}")
    print()
