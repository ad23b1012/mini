"""
MELD dataset loader for multimodal emotion recognition.

This loader supports:
  - face crops extracted from MELD video clips
  - transcript text with optional dialogue-history context
  - cached face-quality assessment for training-time filtering
  - class/sample weights for imbalanced training
"""

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from data.transforms import get_train_transforms, get_val_transforms
from utils.face_quality import check_face_quality


MELD_EMOTIONS = {
    "neutral": 0,
    "surprise": 1,
    "fear": 2,
    "sadness": 3,
    "joy": 4,
    "disgust": 5,
    "anger": 6,
}

MELD_EMOTION_NAMES = {value: key.capitalize() for key, value in MELD_EMOTIONS.items()}

MELD_SENTIMENTS = {
    "neutral": 0,
    "positive": 1,
    "negative": 2,
}


def load_tokenizer(tokenizer_name: str):
    """Load tokenizer fully offline — no HF Hub network calls.

    Setting TRANSFORMERS_OFFLINE=1 suppresses the _patch_mistral_regex
    check inside DeBERTa's tokenizer, which calls model_info() on the Hub
    even when local_files_only=True, crashing when there is no internet.
    """
    import os as _os
    # Block ALL hub calls for this process (harmless if already set)
    _os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    _os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    _os.environ.setdefault("HF_HUB_OFFLINE", "1")
    try:
        return AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=True)
    except OSError:
        # First-time use: temporarily allow network, then lock again
        for _key in ("TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE", "HF_HUB_OFFLINE"):
            _os.environ.pop(_key, None)
        tok = AutoTokenizer.from_pretrained(tokenizer_name)
        _os.environ["TRANSFORMERS_OFFLINE"] = "1"
        _os.environ["HF_DATASETS_OFFLINE"] = "1"
        _os.environ["HF_HUB_OFFLINE"] = "1"
        return tok


class MELDDataset(Dataset):
    """PyTorch dataset for MELD face + text emotion recognition."""

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        image_size: int = 260,
        max_text_length: int = 128,
        text_model_name: str = "microsoft/deberta-v3-base",
        transform=None,
        extract_faces: bool = True,
        num_frames: int = 1,
        use_dialogue_history: bool = False,
        history_window: int = 0,
        include_speaker_in_text: bool = True,
        context_separator: str = " [SEP] ",
        quality_filter: bool = False,
        min_face_quality_score: float = 0.0,
        repair_invalid_faces: bool = False,
        refresh_quality_cache: bool = False,
    ):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.split = split
        self.image_size = image_size
        self.max_text_length = max_text_length
        self.num_frames = num_frames
        self.use_dialogue_history = bool(use_dialogue_history and history_window > 0)
        self.history_window = max(0, history_window)
        self.include_speaker_in_text = include_speaker_in_text
        self.context_separator = context_separator
        self.quality_filter = quality_filter
        self.min_face_quality_score = float(min_face_quality_score)
        self.repair_invalid_faces = repair_invalid_faces
        self.refresh_quality_cache = refresh_quality_cache

        if transform is not None:
            self.transform = transform
        elif split == "train":
            self.transform = get_train_transforms(image_size)
        else:
            self.transform = get_val_transforms(image_size)

        self.tokenizer = load_tokenizer(text_model_name)
        self.samples = self._load_annotations()
        self.dialogue_lookup = self._build_dialogue_lookup()

        if extract_faces:
            self._ensure_face_crops()

        self.face_quality_cache: Dict[str, dict] = {}
        if quality_filter or repair_invalid_faces:
            self.face_quality_cache = self._load_or_build_face_quality_cache()
        if quality_filter:
            self._filter_samples_by_quality()

        self.class_weights = self.get_class_weights(strategy="effective_num")
        self.sample_weights = self._compute_sample_weights()

        print(f"[MELD] Loaded {len(self.samples)} utterances for '{split}' split")

    def _load_annotations(self) -> List[dict]:
        """Load MELD annotations from the official CSV files."""
        split_map = {
            "train": ("train", "train_sent_emo.csv"),
            "dev": ("dev", "dev_sent_emo.csv"),
            "test": ("test", "test_sent_emo.csv"),
        }

        split_dir, csv_name = split_map[self.split]
        csv_path = self.root_dir / split_dir / csv_name
        if not csv_path.exists():
            raise FileNotFoundError(
                f"MELD CSV not found at {csv_path}. "
                "Run `python data/download_datasets.py --dataset meld` first."
            )

        video_dirs = {
            "train": "train_splits",
            "dev": "dev_splits_complete",
            "test": "output_repeated_splits_test",
        }
        video_dir = self.root_dir / split_dir / video_dirs[self.split]
        face_dir = self.root_dir / "face_crops" / self.split

        df = pd.read_csv(csv_path)
        samples = []

        for _, row in df.iterrows():
            utterance = re.sub(r"['\"]", "", str(row.get("Utterance", "")).strip())
            emotion = str(row.get("Emotion", "neutral")).lower().strip()
            sentiment = str(row.get("Sentiment", "neutral")).lower().strip()

            if emotion not in MELD_EMOTIONS:
                continue

            dialogue_id = int(row.get("Dialogue_ID", 0))
            utterance_id = int(row.get("Utterance_ID", 0))
            speaker = str(row.get("Speaker", "Unknown"))
            sample_stem = f"dia{dialogue_id}_utt{utterance_id}"

            samples.append(
                {
                    "sample_key": sample_stem,
                    "utterance": utterance,
                    "emotion": emotion,
                    "emotion_label": MELD_EMOTIONS[emotion],
                    "sentiment": sentiment,
                    "sentiment_label": MELD_SENTIMENTS.get(sentiment, 0),
                    "dialogue_id": dialogue_id,
                    "utterance_id": utterance_id,
                    "speaker": speaker,
                    "video_path": str(video_dir / f"{sample_stem}.mp4"),
                    "face_path": str(face_dir / f"{sample_stem}.jpg"),
                }
            )

        return samples

    def _build_dialogue_lookup(self) -> Dict[int, List[dict]]:
        """Index utterances by dialogue for optional history-aware text inputs."""
        lookup: Dict[int, List[dict]] = {}
        for sample in self.samples:
            lookup.setdefault(sample["dialogue_id"], []).append(sample)
        for dialogue_id in lookup:
            lookup[dialogue_id].sort(key=lambda item: item["utterance_id"])
        return lookup

    def _create_face_detector(self):
        """Create a MediaPipe face detector."""
        try:
            import mediapipe as mp
        except ModuleNotFoundError:
            haar_root = getattr(getattr(cv2, "data", None), "haarcascades", None)
            if not haar_root:
                return None

            cascade = cv2.CascadeClassifier(
                str(Path(haar_root) / "haarcascade_frontalface_default.xml")
            )
            if cascade.empty():
                return None

            return {"kind": "haar", "detector": cascade}

        mp_face_detection = mp.solutions.face_detection
        return {
            "kind": "mediapipe",
            "detector": mp_face_detection.FaceDetection(
                model_selection=0,
                min_detection_confidence=0.3,
            ),
        }

    def _close_face_detector(self, face_detector):
        """Close detector resources when needed."""
        if face_detector is None:
            return

        detector = face_detector
        if isinstance(face_detector, dict):
            detector = face_detector.get("detector")

        close = getattr(detector, "close", None)
        if callable(close):
            close()

    def _build_candidate_positions(self, total_frames: int, dense: bool = False) -> List[int]:
        """Choose frame positions to search for a face crop."""
        if total_frames <= 1:
            return [0]

        if dense:
            positions = np.linspace(0, total_frames - 1, num=min(total_frames, 8), dtype=int)
        else:
            positions = np.array(
                [
                    total_frames // 2,
                    total_frames // 3,
                    (2 * total_frames) // 3,
                    0,
                ],
                dtype=int,
            )

        ordered: List[int] = []
        seen = set()
        for pos in positions.tolist():
            clipped = min(max(int(pos), 0), total_frames - 1)
            if clipped not in seen:
                ordered.append(clipped)
                seen.add(clipped)
        return ordered

    def _extract_best_face_crop(
        self,
        video_path: str,
        face_detector,
        dense_search: bool = False,
    ) -> Optional[np.ndarray]:
        """Extract the best available face crop from a MELD clip."""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            return None

        detector_kind = "none"
        detector = face_detector
        if isinstance(face_detector, dict):
            detector_kind = face_detector.get("kind", "none")
            detector = face_detector.get("detector")
        elif face_detector is not None:
            detector_kind = "mediapipe"

        best_crop = None
        best_score = -1.0

        for frame_pos in self._build_candidate_positions(total_frames, dense=dense_search):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            ih, iw = frame.shape[:2]

            if detector_kind == "mediapipe" and detector is not None:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = detector.process(frame_rgb)
                detections = results.detections if results else None
                if not detections:
                    continue

                for detection in detections:
                    bbox = detection.location_data.relative_bounding_box
                    x = int(bbox.xmin * iw)
                    y = int(bbox.ymin * ih)
                    w = int(bbox.width * iw)
                    h = int(bbox.height * ih)
                    if w <= 0 or h <= 0:
                        continue

                    pad_x = int(0.25 * w)
                    pad_y = int(0.35 * h)
                    x1 = max(0, x - pad_x)
                    y1 = max(0, y - pad_y)
                    x2 = min(iw, x + w + pad_x)
                    y2 = min(ih, y + h + int(0.15 * h))
                    crop = frame[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue

                    area_ratio = (crop.shape[0] * crop.shape[1]) / max(ih * iw, 1)
                    score = float(detection.score[0]) * area_ratio
                    if score > best_score:
                        best_score = score
                        best_crop = crop
            elif detector_kind == "haar" and detector is not None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = detector.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=4,
                    minSize=(max(min(ih, iw) // 6, 24), max(min(ih, iw) // 6, 24)),
                )
                if len(faces) == 0:
                    continue

                for x, y, w, h in faces:
                    pad_x = int(0.2 * w)
                    pad_y = int(0.25 * h)
                    x1 = max(0, x - pad_x)
                    y1 = max(0, y - pad_y)
                    x2 = min(iw, x + w + pad_x)
                    y2 = min(ih, y + h + int(0.1 * h))
                    crop = frame[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue

                    area_ratio = (crop.shape[0] * crop.shape[1]) / max(ih * iw, 1)
                    score = float(area_ratio)
                    if score > best_score:
                        best_score = score
                        best_crop = crop
            else:
                break

        cap.release()

        if best_crop is not None:
            return cv2.resize(best_crop, (self.image_size, self.image_size))

        if dense_search:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2 if total_frames > 0 else 0)
            ret, frame = cap.read()
            cap.release()
            if ret and frame is not None:
                height, width = frame.shape[:2]
                crop_size = min(height, width) * 2 // 3
                cy, cx = height // 2, width // 2
                fallback = frame[
                    max(0, cy - crop_size // 2): cy + crop_size // 2,
                    max(0, cx - crop_size // 2): cx + crop_size // 2,
                ]
                if fallback.size > 0:
                    return cv2.resize(fallback, (self.image_size, self.image_size))

        return None

    def _ensure_face_crops(self):
        """Extract face crops from video clips if missing."""
        face_dir = self.root_dir / "face_crops" / self.split
        face_dir.mkdir(parents=True, exist_ok=True)

        existing = set(os.listdir(face_dir))
        missing = [sample for sample in self.samples if Path(sample["face_path"]).name not in existing]
        if not missing:
            return

        print(f"[MELD] Extracting face crops for {len(missing)} videos...")
        face_detector = self._create_face_detector()
        if face_detector is None:
            print("[MELD] Face detector unavailable; skipping automatic crop extraction.")
            return
        detected_count = 0
        fallback_count = 0

        for sample in missing:
            video_path = sample["video_path"]
            face_path = sample["face_path"]
            if not os.path.exists(video_path):
                continue

            try:
                crop = self._extract_best_face_crop(
                    video_path=video_path,
                    face_detector=face_detector,
                    dense_search=False,
                )
                if crop is None:
                    continue

                if crop.shape[0] == self.image_size and crop.shape[1] == self.image_size:
                    detected_count += 1
                else:
                    fallback_count += 1

                cv2.imwrite(face_path, crop)
            except Exception as exc:
                print(f"  Warning: failed to process {video_path}: {exc}")

        self._close_face_detector(face_detector)
        extracted = len([sample for sample in self.samples if os.path.exists(sample["face_path"])])
        print(
            f"[MELD] Face crops ready: {extracted}/{len(self.samples)} "
            f"(detected: {detected_count}, fallback: {fallback_count})"
        )

    def _load_face_image_uint8(self, face_path: str) -> Optional[np.ndarray]:
        """Load a face crop as uint8 RGB for quality assessment."""
        try:
            return np.asarray(Image.open(face_path).convert("RGB"), dtype=np.uint8)
        except (FileNotFoundError, OSError):
            return None

    def _get_quality_cache_path(self) -> Path:
        """Location for cached face-quality metadata."""
        return self.root_dir / "face_crops" / f"{self.split}_quality_cache.json"

    def _save_quality_cache(self, cache_path: Path, quality_cache: Dict[str, dict]):
        """Write face-quality cache atomically so interrupted runs do not corrupt it."""
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=cache_path.parent,
            delete=False,
            suffix=".tmp",
        ) as handle:
            json.dump(quality_cache, handle, indent=2)
            temp_path = Path(handle.name)

        os.replace(temp_path, cache_path)

    def _sanitize_quality_record(self, quality: dict) -> dict:
        """Convert numpy scalars in quality metadata into JSON-safe Python types."""
        sanitized = {}
        for key, value in quality.items():
            if isinstance(value, np.generic):
                sanitized[key] = value.item()
            else:
                sanitized[key] = value
        return sanitized

    def _repair_face_crop(self, sample: dict, face_detector) -> bool:
        """Try a denser video search to repair an invalid face crop."""
        video_path = sample["video_path"]
        if not os.path.exists(video_path):
            return False

        repaired = self._extract_best_face_crop(
            video_path=video_path,
            face_detector=face_detector,
            dense_search=True,
        )
        if repaired is None:
            return False

        cv2.imwrite(sample["face_path"], repaired)
        return True

    def _load_or_build_face_quality_cache(self) -> Dict[str, dict]:
        """Load cached face quality or build it from the current face crops."""
        cache_path = self._get_quality_cache_path()
        quality_cache: Dict[str, dict] = {}

        if cache_path.exists() and not self.refresh_quality_cache:
            try:
                with open(cache_path, "r", encoding="utf-8") as handle:
                    cached = json.load(handle)
                quality_cache = {key: value for key, value in cached.items()}
            except json.JSONDecodeError:
                print(f"[MELD] Ignoring corrupt face-quality cache at {cache_path}; rebuilding it.")
                quality_cache = {}

        missing_keys = [sample["sample_key"] for sample in self.samples if sample["sample_key"] not in quality_cache]
        should_repair = bool(self.repair_invalid_faces and self.split == "train")
        if not missing_keys and not should_repair:
            return quality_cache

        print(f"[MELD] Building face quality cache for '{self.split}' split...")
        face_detector = self._create_face_detector() if should_repair else None
        if should_repair and face_detector is None:
            print("[MELD] Face repair disabled because no detector backend is available.")
            should_repair = False

        updated = 0
        repaired = 0
        for index, sample in enumerate(self.samples, start=1):
            sample_key = sample["sample_key"]
            cached_quality = quality_cache.get(sample_key)
            already_repair_attempted = bool(
                cached_quality.get("repair_attempted", False)
            ) if cached_quality is not None else False
            if cached_quality is not None and not self.refresh_quality_cache:
                if (
                    not should_repair
                    or cached_quality.get("is_valid", False)
                    or already_repair_attempted
                ):
                    continue

            image = self._load_face_image_uint8(sample["face_path"])
            quality = (
                check_face_quality(image) if image is not None else
                {"is_valid": False, "quality_score": 0.0, "reason": "missing_face_crop"}
            )

            if should_repair and not quality.get("is_valid", False):
                repaired_ok = self._repair_face_crop(sample, face_detector)
                quality["repair_attempted"] = True
                quality["repair_successful"] = bool(repaired_ok)
                if repaired_ok:
                    repaired += 1
                    image = self._load_face_image_uint8(sample["face_path"])
                    quality = (
                        check_face_quality(image) if image is not None else
                        {"is_valid": False, "quality_score": 0.0, "reason": "missing_face_crop"}
                    )
                    quality["repair_attempted"] = True
                    quality["repair_successful"] = True

            quality_cache[sample_key] = self._sanitize_quality_record(quality)
            updated += 1

            if index % 250 == 0:
                print(f"  Assessed {index}/{len(self.samples)} face crops...")
                self._save_quality_cache(cache_path, quality_cache)

        self._close_face_detector(face_detector)

        self._save_quality_cache(cache_path, quality_cache)

        print(f"[MELD] Face quality cache updated ({updated} checked, {repaired} repaired)")
        return quality_cache

    def _filter_samples_by_quality(self):
        """Filter low-quality or invalid face crops from the dataset manifest."""
        before = len(self.samples)
        kept_samples = []
        removed_by_reason: Dict[str, int] = {}

        for sample in self.samples:
            quality = self.face_quality_cache.get(
                sample["sample_key"],
                {"is_valid": False, "quality_score": 0.0, "reason": "missing_quality"},
            )
            is_valid = bool(quality.get("is_valid", False))
            quality_score = float(quality.get("quality_score", 0.0))
            keep = is_valid and quality_score >= self.min_face_quality_score

            if keep:
                sample["face_quality"] = quality
                kept_samples.append(sample)
            else:
                reason = quality.get("reason", "below_min_quality")
                if is_valid and quality_score < self.min_face_quality_score:
                    reason = f"quality_below_threshold ({quality_score:.2f} < {self.min_face_quality_score:.2f})"
                removed_by_reason[reason] = removed_by_reason.get(reason, 0) + 1

        self.samples = kept_samples
        self.dialogue_lookup = self._build_dialogue_lookup()

        after = len(self.samples)
        removed = before - after
        print(
            f"[MELD] Face-quality filtering kept {after}/{before} samples "
            f"(removed {removed})"
        )
        if removed_by_reason:
            for reason, count in sorted(removed_by_reason.items(), key=lambda item: item[1], reverse=True):
                print(f"  - {reason}: {count}")

    def _format_utterance(self, sample: dict) -> str:
        """Create a speaker-aware utterance string."""
        if self.include_speaker_in_text:
            return f"{sample['speaker']}: {sample['utterance']}"
        return sample["utterance"]

    def _build_text_input(self, sample: dict) -> str:
        """Build the text input, optionally with dialogue-history context."""
        current_text = self._format_utterance(sample)
        if not self.use_dialogue_history:
            return current_text

        dialogue = self.dialogue_lookup.get(sample["dialogue_id"], [])
        history = [
            item for item in dialogue
            if item["utterance_id"] < sample["utterance_id"]
        ]
        if self.history_window > 0:
            history = history[-self.history_window:]

        history_text = [self._format_utterance(item) for item in history if item["utterance"]]
        if not history_text:
            return current_text

        return (
            "Dialogue history: "
            + self.context_separator.join(history_text)
            + self.context_separator
            + "Current utterance: "
            + current_text
        )

    def get_class_distribution(self) -> Dict[str, int]:
        """Return class counts for the current filtered manifest."""
        labels = [sample["emotion_label"] for sample in self.samples]
        counts = np.bincount(labels, minlength=len(MELD_EMOTIONS))
        return {
            MELD_EMOTION_NAMES[index]: int(counts[index])
            for index in range(len(MELD_EMOTIONS))
        }

    def get_class_weights(
        self,
        strategy: str = "effective_num",
        beta: float = 0.999,
    ) -> torch.Tensor:
        """Compute normalized class weights from the current manifest."""
        labels = [sample["emotion_label"] for sample in self.samples]
        counts = np.bincount(labels, minlength=len(MELD_EMOTIONS)).astype(np.float32)

        if strategy == "effective_num":
            effective_num = 1.0 - np.power(beta, counts)
            weights = np.zeros_like(counts)
            nonzero = counts > 0
            weights[nonzero] = (1.0 - beta) / np.clip(effective_num[nonzero], 1e-8, None)
        else:
            weights = np.zeros_like(counts)
            nonzero = counts > 0
            weights[nonzero] = 1.0 / counts[nonzero]

        if weights.sum() <= 0:
            weights = np.ones_like(counts)
        weights = weights / weights.sum() * len(MELD_EMOTIONS)
        return torch.tensor(weights, dtype=torch.float32)

    def _compute_sample_weights(self) -> torch.Tensor:
        """Compute per-sample weights for balanced sampling."""
        labels = [sample["emotion_label"] for sample in self.samples]
        counts = np.bincount(labels, minlength=len(MELD_EMOTIONS)).astype(np.float32)
        per_class = 1.0 / np.clip(counts, 1.0, None)
        return torch.tensor([per_class[label] for label in labels], dtype=torch.float32)

    def get_dialogue_context(self, dialogue_id: int) -> List[dict]:
        """Return all utterances for a dialogue, ordered by utterance id."""
        return list(self.dialogue_lookup.get(dialogue_id, []))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        try:
            image = Image.open(sample["face_path"]).convert("RGB")
        except (FileNotFoundError, OSError):
            image = Image.new("RGB", (self.image_size, self.image_size), (128, 128, 128))

        if self.transform:
            image = self.transform(image)

        text_input = self._build_text_input(sample)
        encoded = self.tokenizer(
            text_input,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_length,
            return_tensors="pt",
        )

        quality = self.face_quality_cache.get(sample["sample_key"], sample.get("face_quality", {}))

        return {
            "image": image,
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "label": sample["emotion_label"],
            "sentiment_label": sample["sentiment_label"],
            "utterance": sample["utterance"],
            "text_input": text_input,
            "emotion_name": MELD_EMOTION_NAMES[sample["emotion_label"]],
            "speaker": sample["speaker"],
            "dialogue_id": sample["dialogue_id"],
            "utterance_id": sample["utterance_id"],
            "face_quality_score": float(quality.get("quality_score", 0.0)) if quality else 0.0,
        }
