"""Smoke tests for the passive-liveness (MiniFASNet) preprocessing.

These tests guard against the original bug where ``/255`` normalization was
incorrectly applied to MiniFASNet input, which collapsed real-face scores
to ~0.15 (always-spoof). The model expects raw ``[0, 255]`` float32 inputs.

Both tests skip gracefully when the model file or the ``onnx`` Python
library is unavailable, so they don't break CI environments without models.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

MODEL_FILENAME = "antispoofing.onnx"
INPUT_SIZE = (1, 3, 80, 80)

# Ops that, if present in the model graph, would mean the ONNX export
# already performs preprocessing on the input. The MiniFASNet export
# we ship is a plain ``torch.onnx.export`` with no preprocessing, so
# none of these should appear.
PREPROCESSING_OPS = frozenset({"Div", "Sub", "ReduceMean", "Mean", "Cast", "Identity", "Softmax"})


def _model_path() -> Path | None:
    models_dir = Path(os.environ.get("OPENBIOMETRICS_MODELS_DIR", "./models"))
    candidate = models_dir / MODEL_FILENAME
    return candidate if candidate.exists() else None


@pytest.fixture(scope="module")
def model_path() -> Path:
    path = _model_path()
    if path is None:
        pytest.skip(
            f"Model {MODEL_FILENAME} not found in {os.environ.get('OPENBIOMETRICS_MODELS_DIR', './models')}; "
            "run download_models.py first"
        )
    return path


def test_graph_has_no_input_preprocessing(model_path: Path) -> None:
    """The shipped MiniFASNet export must NOT contain Div/Sub/ReduceMean
    nodes — that would mean the graph itself normalizes input, and the
    consumer should therefore apply ``/255`` before calling it.

    Verified by parsing the actual ONNX protobuf: only Conv, PRelu, Add,
    MatMul, Reshape, and shape-plumbing ops are present.
    """
    try:
        import onnx
    except ImportError:
        pytest.skip("`onnx` Python library not installed; install `engine[dev]`")

    model = onnx.load(str(model_path))
    op_types = {n.op_type for n in model.graph.node}
    leaked = op_types & PREPROCESSING_OPS
    assert not leaked, (
        f"Model graph contains preprocessing ops {leaked}; the model expects "
        "pre-normalized input and the consumer should NOT apply /255."
    )


def test_raw_pixels_produce_nondegenerate_output(model_path: Path) -> None:
    """Feeding ``np.ones * 255`` must produce non-degenerate logits.

    This is the regression test for the original ``/255`` bug: with the
    buggy preprocessing the activations collapsed toward zero and all three
    class probabilities converged to ~1/3, giving is_live ≈ 0.33 regardless
    of input. With correct preprocessing the logits span a meaningful range.
    """
    import onnxruntime as ort

    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    raw = np.ones(INPUT_SIZE, dtype=np.float32) * 255.0
    out = sess.run(None, {input_name: raw})[0]
    spread = float(out.max() - out.min())

    assert spread > 1e-3, (
        f"Raw [0, 255] input collapsed to ~constant output (spread={spread:.6g}); "
        "preprocessing is wrong or model expects a different input range."
    )


def test_model_is_scale_sensitive(model_path: Path) -> None:
    """``np.ones * 255`` and ``np.ones * 1`` must produce different outputs.

    The model is a stack of Convs followed by a MatMul — a purely linear
    network with respect to input magnitude (BN/PRelu aside). If scaled and
    raw inputs produced identical outputs, the graph would be normalizing
    internally and the consumer could safely ``/255`` first. They don't,
    so the consumer must pass raw ``[0, 255]`` floats.
    """
    import onnxruntime as ort

    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    raw = np.ones(INPUT_SIZE, dtype=np.float32) * 255.0
    scaled = np.ones(INPUT_SIZE, dtype=np.float32) * 1.0
    out_raw = sess.run(None, {input_name: raw})[0]
    out_scaled = sess.run(None, {input_name: scaled})[0]

    assert not np.allclose(out_raw, out_scaled, atol=1e-4), (
        "Scaled [0, 1] and raw [0, 255] inputs produced identical output — "
        "the model likely normalizes internally; the consumer-side /255 "
        "preprocessing assumption would be safe to re-introduce."
    )