"""Unified face processing pipeline.

Orchestrates: detect -> quality check -> align -> embed -> liveness -> demographics
Single entry point for all face operations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from openbiometrics.core.detector import DetectedFace, FaceDetector
from openbiometrics.core.demographics import DemographicsEstimator
from openbiometrics.core.liveness import LivenessDetector
from openbiometrics.core.quality import QualityAssessor, QualityReport
from openbiometrics.core.recognizer import FaceRecognizer


@dataclass
class FaceResult:
    """Complete result for a single detected face."""

    face: DetectedFace
    quality: QualityReport | None = None
    embedding: np.ndarray | None = None
    age: int | None = None
    gender: str | None = None
    is_live: bool | None = None
    liveness_score: float | None = None
    identity: str | None = None  # Matched identity from watchlist
    identity_score: float | None = None


@dataclass
class PipelineConfig:
    """Configuration for the face pipeline."""

    models_dir: str = "./models"
    ctx_id: int = 0  # GPU device (-1 for CPU)
    det_thresh: float = 0.5
    det_size: tuple[int, int] = (640, 640)
    max_faces: int = 0
    enable_liveness: bool = True
    enable_demographics: bool = True
    enable_quality: bool = True
    quality_gate: bool = False  # Skip recognition if quality fails

    # Model selection — "auto" picks the best available model on disk
    detector: str = "auto"  # yunet (community) | det_10g (legacy) | auto
    recognizer: str = "auto"  # sface (community) | w600k_r50 (legacy) | auto
    demographics_model: str = "auto"  # vit_genderage (community) | genderage (legacy) | auto


class FacePipeline:
    """End-to-end face processing pipeline.

    Usage:
        pipeline = FacePipeline(PipelineConfig(models_dir="./models"))
        results = pipeline.process(image)
        for r in results:
            print(f"Face: age={r.age}, gender={r.gender}, live={r.is_live}")
            print(f"Embedding shape: {r.embedding.shape}")
    """

    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()
        self._detector: FaceDetector | None = None
        self._recognizer: FaceRecognizer | None = None
        self._liveness: LivenessDetector | None = None
        self._demographics: DemographicsEstimator | None = None
        self._quality = QualityAssessor()

    def load(self) -> None:
        """Load all models. Call once before processing."""
        models = Path(self.config.models_dir)
        ctx = self.config.ctx_id

        # Detector
        det_path = self._resolve_model(
            self.config.detector,
            role="detector",
            preferences=["yunet", "det_10g"],
            models_dir=models,
        )
        if det_path:
            self._detector = FaceDetector(
                model_path=str(det_path),
                ctx_id=ctx,
                det_thresh=self.config.det_thresh,
                det_size=self.config.det_size,
            )
        else:
            # Ultimate fallback: InsightFace FaceAnalysis (requires insightface package)
            self._detector = FaceDetector(
                model_name="buffalo_l",
                ctx_id=ctx,
                det_thresh=self.config.det_thresh,
                det_size=self.config.det_size,
            )

        # Recognizer
        rec_path = self._resolve_model(
            self.config.recognizer,
            role="recognizer",
            preferences=["sface", "w600k_r50"],
            models_dir=models,
        )
        if rec_path:
            self._recognizer = FaceRecognizer(str(rec_path), ctx_id=ctx)

        # Liveness
        if self.config.enable_liveness:
            liv_path = models / "antispoofing.onnx"
            if liv_path.exists():
                self._liveness = LivenessDetector(str(liv_path), ctx_id=ctx)

        # Demographics
        if self.config.enable_demographics:
            dem_path = self._resolve_model(
                self.config.demographics_model,
                role="demographics",
                preferences=["vit_genderage", "genderage"],
                models_dir=models,
            )
            if dem_path:
                self._demographics = DemographicsEstimator(str(dem_path), ctx_id=ctx)

    @property
    def loaded_models(self) -> dict[str, str]:
        """Return a map of role -> loaded model filename for health reporting."""
        result: dict[str, str] = {}
        if self._detector is not None:
            if self._detector._yunet is not None:
                result["detector"] = "yunet"
            elif self._detector._insightface_app is not None:
                result["detector"] = "det_10g"
        if self._recognizer is not None:
            result["recognizer"] = getattr(self._recognizer, "_model", None) and \
                Path(self._recognizer._model._model_path).stem or "unknown"
        if self._demographics is not None:
            result["demographics"] = Path(self._demographics._model._model_path).stem
        if self._liveness is not None:
            result["liveness"] = "antispoofing"
        return result

    @staticmethod
    def _resolve_model(
        selection: str,
        role: str,
        preferences: list[str],
        models_dir: Path,
    ) -> Path | None:
        """Resolve a model selection to a file path.

        Args:
            selection: Model name or "auto"
            role: Functional role (for auto-resolution)
            preferences: Ordered list of model names to try in auto mode
            models_dir: Directory containing model files
        """
        from openbiometrics.runtime.registry import _MODEL_CATALOG

        if selection != "auto":
            info = _MODEL_CATALOG.get(selection)
            if info:
                path = models_dir / info.filename
                if path.exists():
                    return path
            # Direct filename fallback
            direct = models_dir / f"{selection}.onnx"
            if direct.exists():
                return direct
            return None

        # Auto: try preferences in order
        for name in preferences:
            info = _MODEL_CATALOG.get(name)
            if info:
                path = models_dir / info.filename
                if path.exists():
                    return path
        return None

    def process(self, image: np.ndarray) -> list[FaceResult]:
        """Process an image through the full pipeline.

        Args:
            image: BGR numpy array

        Returns:
            List of FaceResult for each detected face
        """
        if self._detector is None:
            raise RuntimeError("Pipeline not loaded. Call pipeline.load() first.")

        faces = self._detector.detect(image, max_faces=self.config.max_faces)
        results = []

        for face in faces:
            result = FaceResult(face=face)

            # Quality assessment
            if self.config.enable_quality:
                result.quality = self._quality.assess(face.aligned, face.landmarks)
                if self.config.quality_gate and not result.quality.is_acceptable:
                    results.append(result)
                    continue

            # Embedding extraction
            if self._recognizer is not None:
                result.embedding = self._recognizer.get_embedding(face.aligned)

            # Liveness
            if self._liveness is not None:
                result.is_live, result.liveness_score = self._liveness.check(face.aligned)

            # Demographics
            if self._demographics is not None:
                result.age, result.gender = self._demographics.estimate(face.aligned)

            results.append(result)

        return results

    def verify(self, image1: np.ndarray, image2: np.ndarray) -> tuple[bool, float]:
        """1:1 verification — do two images show the same person?

        Args:
            image1: BGR image of person A
            image2: BGR image of person B

        Returns:
            (is_match, similarity_score)
        """
        results1 = self.process(image1)
        results2 = self.process(image2)

        if not results1 or not results2:
            return False, 0.0

        if results1[0].embedding is None or results2[0].embedding is None:
            return False, 0.0

        score = FaceRecognizer.compare(results1[0].embedding, results2[0].embedding)
        is_match = FaceRecognizer.compare_to_threshold(score)
        return is_match, score

    def process_file(self, image_path: str) -> list[FaceResult]:
        """Process an image file."""
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        return self.process(image)
