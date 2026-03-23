"""Personal gesture training and KNN inference for KINESYS."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import pickle
from typing import Any

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from config import (
    ACTION_CONFIDENCE_THRESHOLD,
    KNN_K,
    KNN_SAMPLES_REQUIRED,
    LANDMARK_VECTOR_LENGTH,
    MODELS_DIR,
    PERSONAL_MODEL_FILE,
    TRAINER_MIN_SAMPLES_TO_PREDICT,
    TRAINER_SUPPORTED_GESTURES,
)


LOGGER = logging.getLogger(__name__)

MODEL_PAYLOAD_VERSION = 1
EMPTY_PREDICTION_CONFIDENCE = 0.0
SINGLE_VOTE = 1.0


@dataclass(slots=True)
class PredictionResult:
    """One personal-model prediction and its estimated confidence."""

    gesture_name: str | None
    confidence: float
    used_personal_model: bool


class GestureTrainer:
    """Collect landmark samples, train a KNN classifier, and predict personal gestures."""

    def __init__(self, model_path: str = PERSONAL_MODEL_FILE) -> None:
        """Initialize training buffers and load any previously saved personal model."""

        self._model_path = Path(model_path)
        self._samples_by_gesture: dict[str, list[list[float]]] = {}
        self._classifier: KNeighborsClassifier | None = None
        self._label_order: list[str] = []
        self._ensure_models_directory()
        self._load_model()

    def record_sample(
        self,
        gesture_name: str,
        landmarks_norm: list[tuple[float, float, float]],
    ) -> int:
        """Record one sample for a gesture and return the current sample count."""

        self._validate_gesture_name(gesture_name)
        feature_vector = self._flatten_landmarks(landmarks_norm)
        gesture_samples = self._samples_by_gesture.setdefault(gesture_name, [])
        gesture_samples.append(feature_vector)
        return len(gesture_samples)

    def get_sample_count(self, gesture_name: str) -> int:
        """Return the number of recorded samples for one gesture."""

        return len(self._samples_by_gesture.get(gesture_name, []))

    def get_training_progress(self) -> dict[str, int]:
        """Return the current sample counts per gesture."""

        return {gesture_name: len(samples) for gesture_name, samples in self._samples_by_gesture.items()}

    def list_trained_gestures(self) -> list[str]:
        """Return the currently trained gesture labels in stable order."""

        return list(self._label_order)

    def train_gesture(self, gesture_name: str) -> bool:
        """Train or refresh the KNN model after collecting enough samples for one gesture."""

        self._validate_gesture_name(gesture_name)
        if self.get_sample_count(gesture_name) < KNN_SAMPLES_REQUIRED:
            return False
        return self._fit_classifier()

    def train_all(self) -> bool:
        """Train the KNN model from all currently recorded gesture samples."""

        return self._fit_classifier()

    def predict_gesture(
        self,
        landmarks_norm: list[tuple[float, float, float]],
        minimum_confidence: float = ACTION_CONFIDENCE_THRESHOLD,
    ) -> PredictionResult:
        """Predict a gesture using the personal KNN model if enough data exists."""

        if self._classifier is None or not self._label_order:
            return PredictionResult(None, EMPTY_PREDICTION_CONFIDENCE, False)

        feature_vector = np.asarray([self._flatten_landmarks(landmarks_norm)], dtype=np.float32)
        try:
            predicted_label = str(self._classifier.predict(feature_vector)[0])
            neighbor_labels = self._classifier.predict(feature_vector)
            probabilities = self._classifier.predict_proba(feature_vector)[0]
        except Exception as exc:
            LOGGER.exception("Personal gesture prediction failed: %s", exc)
            return PredictionResult(None, EMPTY_PREDICTION_CONFIDENCE, False)

        probability_index = int(np.argmax(probabilities))
        confidence = float(probabilities[probability_index])
        if predicted_label == str(neighbor_labels[0]):
            confidence = max(confidence, SINGLE_VOTE / float(self._classifier.n_neighbors))

        if confidence < minimum_confidence:
            return PredictionResult(None, confidence, True)

        return PredictionResult(predicted_label, confidence, True)

    def delete_gesture(self, gesture_name: str) -> bool:
        """Delete one trained gesture and retrain the remaining KNN model."""

        removed_samples = self._samples_by_gesture.pop(gesture_name, None)
        if removed_samples is None:
            return False

        if self._samples_by_gesture:
            self._fit_classifier()
        else:
            self._classifier = None
            self._label_order = []
            self._save_model()
        return True

    def clear_recorded_samples(self, gesture_name: str | None = None) -> None:
        """Clear unsaved or saved samples either globally or for one gesture."""

        if gesture_name is None:
            self._samples_by_gesture.clear()
            self._classifier = None
            self._label_order = []
            self._save_model()
            return

        self._samples_by_gesture.pop(gesture_name, None)
        if self._samples_by_gesture:
            self._fit_classifier()
        else:
            self.clear_recorded_samples()

    def _fit_classifier(self) -> bool:
        """Fit and persist the KNN model from the current recorded sample set."""

        features: list[list[float]] = []
        labels: list[str] = []

        for gesture_name, gesture_samples in sorted(self._samples_by_gesture.items()):
            if len(gesture_samples) < TRAINER_MIN_SAMPLES_TO_PREDICT:
                continue
            features.extend(gesture_samples)
            labels.extend([gesture_name] * len(gesture_samples))

        if not features or len(set(labels)) == 0:
            self._classifier = None
            self._label_order = []
            self._save_model()
            return False

        neighbors = min(KNN_K, len(features))
        if neighbors <= 0:
            return False

        self._classifier = KNeighborsClassifier(n_neighbors=neighbors, metric="euclidean")
        self._classifier.fit(np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=object))
        self._label_order = sorted(set(labels))
        self._save_model()
        return True

    def _load_model(self) -> None:
        """Load the persisted sample store and trained model if it exists."""

        if not self._model_path.exists():
            return

        try:
            with self._model_path.open("rb") as model_file:
                payload = pickle.load(model_file)
        except Exception as exc:
            LOGGER.exception("Failed to load personal gesture model: %s", exc)
            return

        if not isinstance(payload, dict):
            LOGGER.warning("Unexpected personal model payload format; ignoring persisted model.")
            return

        payload_version = payload.get("version")
        if payload_version != MODEL_PAYLOAD_VERSION:
            LOGGER.warning("Unsupported personal model payload version: %s", payload_version)
            return

        raw_samples = payload.get("samples_by_gesture", {})
        if not isinstance(raw_samples, dict):
            return

        loaded_samples: dict[str, list[list[float]]] = {}
        for gesture_name, gesture_samples in raw_samples.items():
            if not isinstance(gesture_samples, list):
                continue
            valid_samples = [
                list(sample)
                for sample in gesture_samples
                if isinstance(sample, (list, tuple)) and len(sample) == LANDMARK_VECTOR_LENGTH
            ]
            if valid_samples:
                loaded_samples[gesture_name] = valid_samples

        self._samples_by_gesture = loaded_samples
        self._fit_classifier()

    def _save_model(self) -> None:
        """Persist the recorded samples so the trainer can rebuild the model later."""

        payload = {
            "version": MODEL_PAYLOAD_VERSION,
            "samples_by_gesture": self._samples_by_gesture,
        }
        try:
            with self._model_path.open("wb") as model_file:
                pickle.dump(payload, model_file)
        except Exception as exc:
            LOGGER.exception("Failed to save personal gesture model: %s", exc)

    def _ensure_models_directory(self) -> None:
        """Create the models directory if it does not already exist."""

        try:
            Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            LOGGER.exception("Failed to create models directory: %s", exc)

    @staticmethod
    def _flatten_landmarks(landmarks_norm: list[tuple[float, float, float]]) -> list[float]:
        """Flatten 21 xyz landmarks into one 63-float feature vector."""

        feature_vector = [float(value) for point in landmarks_norm for value in point]
        if len(feature_vector) != LANDMARK_VECTOR_LENGTH:
            raise ValueError(
                f"Expected {LANDMARK_VECTOR_LENGTH} landmark values, received {len(feature_vector)}."
            )
        return feature_vector

    @staticmethod
    def _validate_gesture_name(gesture_name: str) -> None:
        """Validate that the requested gesture is supported by the trainer."""

        if gesture_name not in TRAINER_SUPPORTED_GESTURES:
            raise ValueError(f"Unsupported gesture '{gesture_name}'.")
