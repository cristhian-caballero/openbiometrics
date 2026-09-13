"""Central model registry for OpenBiometrics.

Manages model metadata, paths, and downloads for all ONNX models
used across the engine. Ensures models are available before inference.
"""

from __future__ import annotations

import logging
import urllib.request
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_MODELS_DIR = Path("./models")


class ModelTier:
    """Model licensing tiers."""

    COMMUNITY = "community"  # Fully open, commercial use OK
    PREMIUM = "premium"  # Requires API key / license, highest accuracy
    LEGACY = "legacy"  # Non-commercial, kept for backward compatibility


@dataclass(frozen=True)
class ModelInfo:
    """Metadata for a registered model."""

    name: str
    filename: str
    url: str
    description: str
    size_mb: float  # Approximate size in megabytes
    tier: str = ModelTier.COMMUNITY
    license: str = "MIT"
    role: str = ""  # functional role: detector, recognizer, demographics, liveness, etc.


# Model catalog — covers current and planned models.
# URLs point to publicly hosted ONNX files (OpenCV Zoo, HuggingFace, etc.)
_MODEL_CATALOG: dict[str, ModelInfo] = {
    # ── Community tier (open license, default) ─────────────────────────
    "yunet": ModelInfo(
        name="yunet",
        filename="yunet.onnx",
        url="https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        description="YuNet face detector — 75K params, 0.88 AP WIDER Face",
        size_mb=0.3,
        tier=ModelTier.COMMUNITY,
        license="MIT",
        role="detector",
    ),
    "sface": ModelInfo(
        name="sface",
        filename="sface.onnx",
        url="https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
        description="SFace face recognition — 99.4% LFW, MobileFaceNet",
        size_mb=37.0,
        tier=ModelTier.COMMUNITY,
        license="Apache-2.0",
        role="recognizer",
    ),
    "vit_genderage": ModelInfo(
        name="vit_genderage",
        filename="vit_genderage.onnx",
        url="https://huggingface.co/onnx-community/age-gender-prediction-ONNX/resolve/main/onnx/model.onnx",
        description="ViT-base age/gender — 94.3% gender acc, 4.5yr age MAE",
        size_mb=330.0,
        tier=ModelTier.COMMUNITY,
        license="Apache-2.0",
        role="demographics",
    ),
    "antispoofing": ModelInfo(
        name="antispoofing",
        filename="antispoofing.onnx",
        # NB: the HF model card's "Preprocessing" section (which says
        # `pixel / 255`) describes the upstream PyTorch training data
        # loader, NOT what's baked into the ONNX graph. The graph has
        # no input preprocessing; pass raw [0, 255] float32 BGR pixels.
        # See LivenessDetector._preprocess and tests/test_liveness_preprocess.py.
        url="https://huggingface.co/garciafido/minifasnet-v2-anti-spoofing-onnx/resolve/main/minifasnet_v2.onnx",
        description="MiniFASNet passive liveness / anti-spoofing",
        size_mb=1.7,
        tier=ModelTier.COMMUNITY,
        license="Apache-2.0",
        role="liveness",
    ),
    "yolov8n": ModelInfo(
        name="yolov8n",
        filename="yolov8n.pt",
        url="https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt",
        description="YOLOv8 nano for person/object detection",
        size_mb=6.5,
        tier=ModelTier.COMMUNITY,
        license="AGPL-3.0",
        role="person_detector",
    ),
    "face_mesh": ModelInfo(
        name="face_mesh",
        filename="face_mesh.onnx",
        url="https://huggingface.co/yakhyo/uniface-weights/resolve/main/face_mesh.onnx",
        description="MediaPipe face mesh 468-point landmarks",
        size_mb=2.4,
        tier=ModelTier.COMMUNITY,
        license="Apache-2.0",
        role="landmarks",
    ),
    # ── Premium tier (highest accuracy, requires license key) ──────────
    # These are placeholders — actual premium models would be hosted on
    # a private registry and require authentication to download.
    # Partners (e.g. Innovatrics) can register their own models here.

    # ── Legacy tier (non-commercial, kept for backward compat) ─────────
    "det_10g": ModelInfo(
        name="det_10g",
        filename="det_10g.onnx",
        url="https://huggingface.co/deepghs/insightface/resolve/main/buffalo_l/det_10g.onnx",
        description="SCRFD 10G face detector (InsightFace, non-commercial)",
        size_mb=16.9,
        tier=ModelTier.LEGACY,
        license="non-commercial",
        role="detector",
    ),
    "w600k_r50": ModelInfo(
        name="w600k_r50",
        filename="w600k_r50.onnx",
        url="https://huggingface.co/deepghs/insightface/resolve/main/buffalo_l/w600k_r50.onnx",
        description="ArcFace ResNet-50 face recognition (InsightFace, non-commercial)",
        size_mb=166.0,
        tier=ModelTier.LEGACY,
        license="non-commercial",
        role="recognizer",
    ),
    "genderage": ModelInfo(
        name="genderage",
        filename="genderage.onnx",
        url="https://huggingface.co/deepghs/insightface/resolve/main/buffalo_l/genderage.onnx",
        description="InsightFace age/gender (non-commercial, lower accuracy)",
        size_mb=1.3,
        tier=ModelTier.LEGACY,
        license="non-commercial",
        role="demographics",
    ),
}


class ModelRegistry:
    """Central registry for managing biometric model files.

    Provides a single place to query model metadata, check availability,
    and download missing models.

    Usage:
        registry = ModelRegistry(models_dir="./models")
        rec_path = registry.ensure_model("w600k_r50")
        # rec_path is now guaranteed to exist on disk
    """

    def __init__(self, models_dir: str | Path = _DEFAULT_MODELS_DIR):
        """
        Args:
            models_dir: Directory where model files are stored / downloaded to
        """
        self._models_dir = Path(models_dir)

    @property
    def models_dir(self) -> Path:
        """Base directory for model storage."""
        return self._models_dir

    def ensure_model(self, name: str) -> Path:
        """Ensure a model is available on disk, downloading if necessary.

        Args:
            name: Model name from the catalog (e.g. "w600k_r50")

        Returns:
            Absolute path to the model file

        Raises:
            KeyError: If model name is not in the catalog
            RuntimeError: If download fails
        """
        info = self._get_info(name)
        path = self._models_dir / info.filename

        if path.exists():
            logger.debug("Model '%s' found at %s", name, path)
            return path

        logger.info("Model '%s' not found locally, downloading from %s", name, info.url)
        self._download(info.url, path)
        return path

    def model_path(self, name: str) -> Path:
        """Get the expected local path for a model (without downloading).

        Args:
            name: Model name from the catalog

        Returns:
            Expected path (may or may not exist)
        """
        info = self._get_info(name)
        return self._models_dir / info.filename

    def is_available(self, name: str) -> bool:
        """Check if a model file exists on disk.

        Args:
            name: Model name from the catalog
        """
        info = self._get_info(name)
        return (self._models_dir / info.filename).exists()

    def list_models(self) -> list[ModelInfo]:
        """Return metadata for all registered models."""
        return list(_MODEL_CATALOG.values())

    def register(self, info: ModelInfo) -> None:
        """Register a custom model in the catalog.

        Args:
            info: ModelInfo for the new model. Overwrites if name already exists.
        """
        _MODEL_CATALOG[info.name] = info

    def _get_info(self, name: str) -> ModelInfo:
        if name not in _MODEL_CATALOG:
            available = ", ".join(sorted(_MODEL_CATALOG.keys()))
            raise KeyError(f"Unknown model '{name}'. Available: {available}")
        return _MODEL_CATALOG[name]

    def _download(self, url: str, dest: Path) -> None:
        """Download a file from url to dest."""
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".part")
        try:
            logger.info("Downloading %s -> %s", url, dest)
            urllib.request.urlretrieve(url, str(tmp))
            tmp.rename(dest)
            logger.info("Download complete: %s (%.1f MB)", dest.name, dest.stat().st_size / 1e6)
        except Exception as exc:
            if tmp.exists():
                tmp.unlink()
            raise RuntimeError(f"Failed to download model from {url}: {exc}") from exc
