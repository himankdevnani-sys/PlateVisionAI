"""
src/detector.py — YOLOv8-based number plate detector.

Author: Abhayvir Singh (Team Lead) — YOLOv8 integration and plate detection.

This module wraps the Ultralytics YOLO API to detect number-plate regions.
A plate-specific model (not a generic COCO detector) must be supplied via
config.MODEL_PATH.  See models/README.md for weight acquisition instructions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

import config
from src.preprocessing import preprocess_for_detection

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class PlateDetection:
    """A single detected number plate."""

    #: Bounding box in pixel coordinates (x1, y1, x2, y2), clipped to image.
    bbox: Tuple[int, int, int, int]

    #: Detector confidence score (0.0–1.0).
    confidence: float

    #: Class label returned by the model (e.g. "license_plate") or "" if unknown.
    class_name: str = ""

    #: Cropped plate image (BGR numpy array). Populated by plate_extractor.
    crop: Optional[np.ndarray] = field(default=None, repr=False)

    def area(self) -> int:
        """Return bounding box area in pixels²."""
        x1, y1, x2, y2 = self.bbox
        return max(0, x2 - x1) * max(0, y2 - y1)


@dataclass
class DetectionResult:
    """Result returned by the detector for one image."""

    #: List of detected plates (may be empty).
    plates: List[PlateDetection] = field(default_factory=list)

    #: True when the model ran successfully (even if no plates were found).
    success: bool = True

    #: Human-readable error message (empty on success).
    error_message: str = ""

    @property
    def plate_count(self) -> int:
        return len(self.plates)

    @property
    def best_plate(self) -> Optional[PlateDetection]:
        """Return the plate with the highest confidence, or None."""
        if not self.plates:
            return None
        return max(self.plates, key=lambda p: p.confidence)


# ---------------------------------------------------------------------------
# Configuration error
# ---------------------------------------------------------------------------


class ConfigurationError(RuntimeError):
    """Raised when model weights are missing or the model fails to load."""


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


class PlateDetector:
    """YOLOv8-based licence-plate detector.

    Usage::

        detector = PlateDetector()
        result = detector.detect(bgr_image)
        for plate in result.plates:
            print(plate.confidence, plate.bbox)

    The detector is initialised lazily so that importing this module does
    not trigger a model load.  Call :meth:`load_model` explicitly, or let
    :meth:`detect` do it on the first call.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        conf_threshold: float = config.DETECTION_CONFIDENCE,
        img_size: int = config.INFERENCE_IMG_SIZE,
        device: str = config.DEVICE,
    ) -> None:
        """Initialise detector parameters (does not load model yet).

        Args:
            model_path: Path to .pt weight file.  Defaults to config.MODEL_PATH.
            conf_threshold: Minimum confidence to report a detection (0–1).
            img_size: YOLO inference image size (pixels, multiple of 32).
            device: Inference device string ("cpu", "cuda", ...).
        """
        self.model_path = model_path or config.MODEL_PATH
        self.conf_threshold = conf_threshold
        self.img_size = img_size
        self.device = device
        self._model = None  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """Load the YOLO model from disk.

        Raises:
            ConfigurationError: If the weight file does not exist or the model
                cannot be initialised.
        """
        from ultralytics import YOLO

        weight_path = Path(self.model_path)
        if not weight_path.exists():
            raise ConfigurationError(
                f"Model weight file not found: {weight_path}\n"
                "Please download a licence-plate YOLOv8 model and place it at "
                f"'{weight_path}' or set the ALPR_MODEL_PATH environment variable.\n"
                "See models/README.md for instructions."
            )

        logger.info("Loading YOLO model from %s (device=%s)", weight_path, self.device)
        try:
            self._model = YOLO(str(weight_path))
            logger.info("YOLO model loaded successfully.")
        except Exception as exc:
            raise ConfigurationError(
                f"Failed to load YOLO model from '{weight_path}': {exc}"
            ) from exc

    @property
    def is_loaded(self) -> bool:
        """True when the model is ready for inference."""
        return self._model is not None

    def ensure_loaded(self) -> None:
        """Load the model if it has not been loaded yet."""
        if not self.is_loaded:
            self.load_model()

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def detect(self, image: np.ndarray) -> DetectionResult:
        """Run plate detection on a BGR image.

        Args:
            image: Full-frame vehicle image as a BGR numpy array.

        Returns:
            A :class:`DetectionResult` containing all detected plates.
        """
        try:
            self.ensure_loaded()
        except ConfigurationError as exc:
            logger.error("Detector not configured: %s", exc)
            return DetectionResult(success=False, error_message=str(exc))

        # Preprocess (resize large images to avoid OOM)
        processed = preprocess_for_detection(image)
        h, w = processed.shape[:2]

        try:
            results = self._model.predict(
                source=processed,
                conf=self.conf_threshold,
                imgsz=self.img_size,
                device=self.device,
                verbose=False,
            )
        except Exception as exc:
            logger.exception("YOLO inference failed: %s", exc)
            return DetectionResult(
                success=False,
                error_message=f"Detection inference failed: {exc}",
            )

        plates: List[PlateDetection] = []
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            # Scale factor if image was resized before inference
            orig_h, orig_w = image.shape[:2]
            scale_x = orig_w / w
            scale_y = orig_h / h

            for i, box in enumerate(boxes):
                conf = float(box.conf[0])
                if conf < self.conf_threshold:
                    continue

                # Bounding box in inference-image coordinates
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                # Map back to original image coordinates
                x1 = int(x1 * scale_x)
                y1 = int(y1 * scale_y)
                x2 = int(x2 * scale_x)
                y2 = int(y2 * scale_y)

                # Clip to original image boundaries
                x1 = max(0, min(x1, orig_w - 1))
                y1 = max(0, min(y1, orig_h - 1))
                x2 = max(x1 + 1, min(x2, orig_w))
                y2 = max(y1 + 1, min(y2, orig_h))

                class_name = ""
                if result.names and box.cls is not None:
                    class_idx = int(box.cls[0])
                    raw_name = result.names.get(class_idx, "")
                    # Some HuggingFace models store names as "['plate']" — clean that
                    class_name = str(raw_name).strip("[]'\" ").split(",")[0].strip().lower() or "license_plate"

                plates.append(
                    PlateDetection(
                        bbox=(x1, y1, x2, y2),
                        confidence=conf,
                        class_name=class_name,
                    )
                )

        # Sort by confidence (descending) then by x-coordinate (left to right)
        plates.sort(key=lambda p: (-p.confidence, p.bbox[0]))

        logger.info("Detected %d plate(s).", len(plates))
        return DetectionResult(plates=plates)
