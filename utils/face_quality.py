"""
Face quality assessment for training and XAI.

The preferred path uses MediaPipe FaceMesh for strong quality checks. When
MediaPipe is unavailable, the code falls back to OpenCV Haar cascades so the
pipeline can still filter obvious bad crops instead of crashing.
"""

import os
from functools import lru_cache

import cv2
import numpy as np


@lru_cache(maxsize=1)
def _get_mediapipe():
    """Return the mediapipe module if available, otherwise None."""
    try:
        import mediapipe as mp
    except ModuleNotFoundError:
        return None
    return mp


@lru_cache(maxsize=2)
def _load_haar_cascade(filename: str):
    """Load an OpenCV Haar cascade if present on this machine."""
    haar_root = getattr(getattr(cv2, "data", None), "haarcascades", None)
    if not haar_root:
        return None

    cascade_path = os.path.join(haar_root, filename)
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        return None
    return cascade


def _check_face_quality_haar(
    image: np.ndarray,
    min_face_ratio: float,
) -> dict:
    """Fallback quality estimation using OpenCV Haar cascades."""
    h, w = image.shape[:2]
    result = {
        "is_valid": False,
        "face_ratio": 0.0,
        "num_landmarks": 0,
        "is_frontal": False,
        "eyes_visible": False,
        "reason": "",
        "quality_score": 0.0,
        "backend": "haar",
    }

    face_cascade = _load_haar_cascade("haarcascade_frontalface_default.xml")
    if face_cascade is None:
        result["reason"] = "face_backend_unavailable"
        return result

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    min_size = max(min(h, w) // 5, 24)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(min_size, min_size),
    )
    if len(faces) == 0:
        result["reason"] = "no_face_detected"
        return result

    x, y, fw, fh = max(faces, key=lambda box: box[2] * box[3])
    face_ratio = float((fw * fh) / max(h * w, 1))
    result["face_ratio"] = face_ratio

    if face_ratio < min_face_ratio:
        result["reason"] = f"face_too_small ({face_ratio:.2f} < {min_face_ratio})"
        return result

    aspect_ratio = fw / max(fh, 1)
    is_frontal = 0.65 <= aspect_ratio <= 1.65
    result["is_frontal"] = is_frontal

    eye_cascade = _load_haar_cascade("haarcascade_eye.xml")
    eyes_visible = True
    if eye_cascade is not None:
        face_roi = gray[y:y + fh, x:x + fw]
        eyes = eye_cascade.detectMultiScale(
            face_roi,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(max(fw // 10, 8), max(fh // 10, 8)),
        )
        eyes_visible = len(eyes) >= 1
    result["eyes_visible"] = eyes_visible

    score = 0.0
    score += 0.65 * min(face_ratio / 0.45, 1.0)
    score += 0.2 * (1.0 if is_frontal else 0.0)
    score += 0.15 * (1.0 if eyes_visible else 0.0)

    result["quality_score"] = float(min(score, 1.0))
    result["is_valid"] = True
    result["reason"] = ""
    return result


def _check_face_quality_mediapipe(
    image: np.ndarray,
    min_face_ratio: float,
    min_landmarks: int,
    mp,
) -> dict:
    """High-fidelity face quality assessment using MediaPipe FaceMesh."""
    h, w = image.shape[:2]
    result = {
        "is_valid": False,
        "face_ratio": 0.0,
        "num_landmarks": 0,
        "is_frontal": False,
        "eyes_visible": False,
        "reason": "",
        "quality_score": 0.0,
        "backend": "mediapipe",
    }

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.3,
    )

    mesh_results = face_mesh.process(image)
    face_mesh.close()

    if not mesh_results.multi_face_landmarks:
        result["reason"] = "no_face_detected"
        return result

    landmarks = mesh_results.multi_face_landmarks[0]
    num_landmarks = len(landmarks.landmark)
    result["num_landmarks"] = num_landmarks

    if num_landmarks < min_landmarks:
        result["reason"] = f"insufficient_landmarks ({num_landmarks}/{min_landmarks})"
        return result

    xs = [lm.x for lm in landmarks.landmark]
    ys = [lm.y for lm in landmarks.landmark]

    face_x_min = max(0.0, min(xs))
    face_x_max = min(1.0, max(xs))
    face_y_min = max(0.0, min(ys))
    face_y_max = min(1.0, max(ys))

    face_width = face_x_max - face_x_min
    face_height = face_y_max - face_y_min
    face_ratio = face_width * face_height
    result["face_ratio"] = float(face_ratio)

    if face_ratio < min_face_ratio:
        result["reason"] = f"face_too_small ({face_ratio:.2f} < {min_face_ratio})"
        return result

    nose_tip = landmarks.landmark[1]
    left_ear = landmarks.landmark[234]
    right_ear = landmarks.landmark[454]

    nose_x = nose_tip.x
    left_ear_x = left_ear.x
    right_ear_x = right_ear.x
    ear_min_x = min(left_ear_x, right_ear_x)
    ear_max_x = max(left_ear_x, right_ear_x)
    ear_span = ear_max_x - ear_min_x

    is_frontal = (ear_min_x - 0.05) <= nose_x <= (ear_max_x + 0.05)
    if ear_span < 0.08:
        is_frontal = False

    result["is_frontal"] = is_frontal
    if not is_frontal:
        result["reason"] = "profile_view"
        return result

    left_eye_indices = [33, 133, 159, 145]
    right_eye_indices = [362, 263, 386, 374]

    def eye_is_visible(indices):
        pts = []
        for idx in indices:
            lm = landmarks.landmark[idx]
            pts.append([lm.x * w, lm.y * h])
        pts = np.array(pts)
        eye_w = pts[:, 0].max() - pts[:, 0].min()
        eye_h = pts[:, 1].max() - pts[:, 1].min()
        return eye_w > 3 and eye_h > 1

    left_eye_ok = eye_is_visible(left_eye_indices)
    right_eye_ok = eye_is_visible(right_eye_indices)
    eyes_visible = left_eye_ok and right_eye_ok
    result["eyes_visible"] = eyes_visible

    if not eyes_visible:
        result["reason"] = "eyes_not_visible"
        return result

    score = 0.0
    score += 0.3 * min(face_ratio / 0.5, 1.0)
    score += 0.3 * (1.0 if is_frontal else 0.0)
    score += 0.2 * min(num_landmarks / 468, 1.0)
    score += 0.1 * (1.0 if left_eye_ok else 0.0)
    score += 0.1 * (1.0 if right_eye_ok else 0.0)

    if ear_span > 0:
        nose_center_ratio = abs((nose_x - ear_min_x) / ear_span - 0.5)
        score += 0.05 * max(0, 1.0 - nose_center_ratio * 2)

    result["quality_score"] = float(min(score, 1.0))
    result["is_valid"] = True
    result["reason"] = ""
    return result


def check_face_quality(
    image: np.ndarray,
    min_face_ratio: float = 0.20,
    min_landmarks: int = 400,
) -> dict:
    """
    Assess whether a face crop image is suitable for training and XAI.

    Args:
        image: RGB image [H, W, 3] as uint8.
        min_face_ratio: Minimum fraction of image area the face must occupy.
        min_landmarks: Minimum number of landmarks required by the MediaPipe path.

    Returns:
        Dict containing validity, heuristics, and a normalized quality score.
    """
    if image.dtype != np.uint8:
        image = np.uint8(np.clip(image, 0, 255))

    mp = _get_mediapipe()
    if mp is not None:
        return _check_face_quality_mediapipe(
            image=image,
            min_face_ratio=min_face_ratio,
            min_landmarks=min_landmarks,
            mp=mp,
        )

    return _check_face_quality_haar(
        image=image,
        min_face_ratio=min_face_ratio,
    )


def get_face_quality_summary(quality: dict) -> str:
    """Format a face quality result as a human-readable string."""
    backend = quality.get("backend")
    backend_suffix = f" [{backend}]" if backend else ""
    if quality["is_valid"]:
        return (
            f"[OK]{backend_suffix} Quality={quality['quality_score']:.2f} "
            f"(ratio={quality['face_ratio']:.2f}, "
            f"landmarks={quality['num_landmarks']}, "
            f"frontal={quality['is_frontal']})"
        )
    return f"[SKIP]{backend_suffix} REJECTED: {quality['reason']}"
