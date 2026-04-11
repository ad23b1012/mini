"""
MELD (Multimodal EmotionLines Dataset) Loader.

MELD is a multimodal multi-party emotional dialogue dataset from the
TV series "Friends". Each utterance has:
  - Video clip (face + scene)
  - Transcript text
  - Emotion label: neutral, surprise, fear, sadness, joy, disgust, anger
  - Sentiment label: positive, negative, neutral

Citation:
    Poria, S., Hazarika, D., Majumder, N., Naik, G., Cambria, E. and
    Mihalcea, R., 2019. MELD: A Multimodal Multi-Party Dataset for
    Emotion Recognition in Conversations. ACL 2019.

Download: https://github.com/declare-lab/MELD
"""

import os
import re
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from data.transforms import get_train_transforms, get_val_transforms


# MELD emotion label mapping
MELD_EMOTIONS = {
    "neutral": 0,
    "surprise": 1,
    "fear": 2,
    "sadness": 3,
    "joy": 4,
    "disgust": 5,
    "anger": 6,
}

MELD_EMOTION_NAMES = {v: k.capitalize() for k, v in MELD_EMOTIONS.items()}

# Sentiment mapping
MELD_SENTIMENTS = {
    "neutral": 0,
    "positive": 1,
    "negative": 2,
}


class MELDDataset(Dataset):
    """
    PyTorch Dataset for MELD — multimodal emotion recognition.

    Loads face frames extracted from video + transcript text for each utterance.

    Expected directory structure after download + preprocessing:
        meld/
        ├── train/
        │   ├── train_sent_emo.csv
        │   └── train_splits/         # Video files: dia{X}_utt{Y}.mp4
        ├── dev/
        │   ├── dev_sent_emo.csv
        │   └── dev_splits_complete/
        ├── test/
        │   ├── test_sent_emo.csv
        │   └── output_repeated_splits_test/
        └── face_crops/               # Extracted face frames (created by preprocess)
            ├── train/
            ├── dev/
            └── test/
    """

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
    ):
        """
        Args:
            root_dir: Path to MELD root directory.
            split: "train", "dev", or "test".
            image_size: Target face crop size.
            max_text_length: Maximum token length for text encoder.
            text_model_name: HuggingFace tokenizer name.
            transform: Optional image transforms.
            extract_faces: Whether to extract face crops from video on first run.
            num_frames: Number of frames to sample from each video clip.
        """
        super().__init__()
        self.root_dir = Path(root_dir)
        self.split = split
        self.image_size = image_size
        self.max_text_length = max_text_length
        self.num_frames = num_frames

        # Set transforms
        if transform is not None:
            self.transform = transform
        elif split == "train":
            self.transform = get_train_transforms(image_size)
        else:
            self.transform = get_val_transforms(image_size)

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)

        # Load annotations
        self.samples = self._load_annotations()

        # Extract face crops if needed
        if extract_faces:
            self._ensure_face_crops()

        print(
            f"[MELD] Loaded {len(self.samples)} utterances for '{split}' split"
        )

    def _load_annotations(self) -> list:
        """Load MELD CSV annotations."""
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

        df = pd.read_csv(csv_path)

        # Video directory mapping
        video_dirs = {
            "train": "train_splits",
            "dev": "dev_splits_complete",
            "test": "output_repeated_splits_test",
        }
        video_dir = self.root_dir / split_dir / video_dirs[self.split]

        samples = []
        for _, row in df.iterrows():
            # Clean utterance text
            utterance = str(row.get("Utterance", "")).strip()
            utterance = re.sub(r"['\"]", "", utterance)  # Remove quotes

            emotion = str(row.get("Emotion", "neutral")).lower().strip()
            sentiment = str(row.get("Sentiment", "neutral")).lower().strip()

            if emotion not in MELD_EMOTIONS:
                continue

            # Dialogue and utterance IDs
            dia_id = int(row.get("Dialogue_ID", 0))
            utt_id = int(row.get("Utterance_ID", 0))

            # Video file path
            video_name = f"dia{dia_id}_utt{utt_id}.mp4"
            video_path = video_dir / video_name

            # Face crop path (to be extracted)
            face_dir = self.root_dir / "face_crops" / self.split
            face_path = face_dir / f"dia{dia_id}_utt{utt_id}.jpg"

            # Speaker info (for context)
            speaker = str(row.get("Speaker", "Unknown"))

            samples.append(
                {
                    "utterance": utterance,
                    "emotion": emotion,
                    "emotion_label": MELD_EMOTIONS[emotion],
                    "sentiment": sentiment,
                    "sentiment_label": MELD_SENTIMENTS.get(sentiment, 0),
                    "dialogue_id": dia_id,
                    "utterance_id": utt_id,
                    "speaker": speaker,
                    "video_path": str(video_path),
                    "face_path": str(face_path),
                }
            )

        return samples

    def _ensure_face_crops(self):
        """Extract face crops from video clips if not already done."""
        face_dir = self.root_dir / "face_crops" / self.split
        face_dir.mkdir(parents=True, exist_ok=True)

        # Check if face crops already exist
        existing = set(os.listdir(face_dir))
        missing = [
            s for s in self.samples
            if Path(s["face_path"]).name not in existing
        ]

        if not missing:
            return

        print(f"[MELD] Extracting face crops for {len(missing)} videos...")

        # Load OpenCV face detector
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        face_cascade = cv2.CascadeClassifier(cascade_path)

        for sample in missing:
            video_path = sample["video_path"]
            face_path = sample["face_path"]

            if not os.path.exists(video_path):
                continue

            try:
                cap = cv2.VideoCapture(video_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                if total_frames == 0:
                    cap.release()
                    continue

                # Sample the middle frame
                mid_frame = total_frames // 2
                cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
                ret, frame = cap.read()
                cap.release()

                if not ret or frame is None:
                    continue

                # Detect face
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
                )

                if len(faces) > 0:
                    # Use the largest face
                    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                    # Add padding (20%)
                    pad = int(0.2 * max(w, h))
                    y1 = max(0, y - pad)
                    y2 = min(frame.shape[0], y + h + pad)
                    x1 = max(0, x - pad)
                    x2 = min(frame.shape[1], x + w + pad)
                    face_crop = frame[y1:y2, x1:x2]
                else:
                    # No face detected — use center crop
                    h, w = frame.shape[:2]
                    crop_size = min(h, w) // 2
                    cy, cx = h // 2, w // 2
                    face_crop = frame[
                        cy - crop_size:cy + crop_size,
                        cx - crop_size:cx + crop_size,
                    ]

                # Resize and save
                face_crop = cv2.resize(
                    face_crop, (self.image_size, self.image_size)
                )
                cv2.imwrite(face_path, face_crop)

            except Exception as e:
                print(f"  Warning: Failed to process {video_path}: {e}")

        extracted = len([s for s in self.samples if os.path.exists(s["face_path"])])
        print(f"[MELD] Face crops ready: {extracted}/{len(self.samples)}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        """
        Returns:
            dict with keys:
                - image: Tensor [3, H, W] — face crop
                - input_ids: Tensor [max_len] — tokenized text
                - attention_mask: Tensor [max_len]
                - label: int (emotion class index)
                - sentiment_label: int
                - utterance: str (raw text)
                - emotion_name: str
                - speaker: str
                - dialogue_id: int
                - utterance_id: int
        """
        sample = self.samples[idx]

        # ---- Load face image ----
        face_path = sample["face_path"]
        try:
            image = Image.open(face_path).convert("RGB")
        except (FileNotFoundError, OSError):
            # Fallback: black image
            image = Image.new("RGB", (self.image_size, self.image_size), (0, 0, 0))

        if self.transform:
            image = self.transform(image)

        # ---- Tokenize text ----
        encoded = self.tokenizer(
            sample["utterance"],
            padding="max_length",
            truncation=True,
            max_length=self.max_text_length,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)

        return {
            "image": image,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": sample["emotion_label"],
            "sentiment_label": sample["sentiment_label"],
            "utterance": sample["utterance"],
            "emotion_name": MELD_EMOTION_NAMES[sample["emotion_label"]],
            "speaker": sample["speaker"],
            "dialogue_id": sample["dialogue_id"],
            "utterance_id": sample["utterance_id"],
        }

    def get_class_distribution(self) -> dict:
        """Return class distribution as {emotion_name: count}."""
        labels = [s["emotion_label"] for s in self.samples]
        counts = np.bincount(labels, minlength=len(MELD_EMOTIONS))
        return {
            MELD_EMOTION_NAMES[i]: int(counts[i])
            for i in range(len(MELD_EMOTIONS))
        }

    def get_dialogue_context(self, dialogue_id: int) -> list:
        """Get all utterances in a dialogue, ordered by utterance_id."""
        dialogue = [
            s for s in self.samples if s["dialogue_id"] == dialogue_id
        ]
        return sorted(dialogue, key=lambda x: x["utterance_id"])
