from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


@dataclass
class RoboflowPreprocessor:
    model: Any
    class_name: str = "tote-bin"
    pad: int = 10
    bg: int = 0

    def __call__(self, pil_image: Image.Image, image_ref: str) -> Image.Image:
        del image_ref  # Reserved for logging/debugging hooks.

        image_np = np.array(pil_image.convert("RGB"))
        height, width = image_np.shape[:2]

        inference_result = self.model.infer(image_np)
        predictions = [p.json() for p in getattr(inference_result, "predictions", [])]
        mask, _meta = get_best_bin_mask({"predictions": predictions}, height, width, self.class_name)
        cropped = crop_mask_pad_square(pil_image, mask, pad=self.pad, bg=self.bg)
        if cropped is None:
            return pil_image
        roi_image, _bbox = cropped
        return roi_image


_OPTIONAL_INFERENCE_MODEL_FLAGS = (
    "CORE_MODEL_SAM_ENABLED",
    "CORE_MODEL_SAM3_ENABLED",
    "CORE_MODEL_GAZE_ENABLED",
    "CORE_MODEL_YOLO_WORLD_ENABLED",
)
_CUDA_PROVIDER = "CUDAExecutionProvider"


def _set_inference_optional_model_defaults() -> None:
    for env_name in _OPTIONAL_INFERENCE_MODEL_FLAGS:
        os.environ.setdefault(env_name, "False")


def _log_onnx_runtime_provider_status() -> None:
    try:
        import onnxruntime as ort
    except ImportError:
        print("ONNX Runtime not importable. Install project extra `.[roboflow-gpu]` for CUDA support.")
        return

    providers = ort.get_available_providers()
    if _CUDA_PROVIDER in providers:
        print(f"ONNX Runtime providers: {providers} (CUDA enabled).")
        return

    print(
        "ONNX Runtime providers: "
        f"{providers} (CUDA unavailable). "
        "Install project extra `.[roboflow-gpu]` and remove `inference` / `onnxruntime` CPU-only packages."
    )


def build_roboflow_preprocessor(
    roboflow_model_id: str,
    roboflow_api_key: str,
    class_name: str,
    pad: int,
    bg: int,
) -> RoboflowPreprocessor:
    _set_inference_optional_model_defaults()

    try:
        from inference import get_roboflow_model
    except ImportError as exc:
        raise RuntimeError(
            "Roboflow preprocessing requires Roboflow Inference. "
            "Install with: pip install -e \".[roboflow-gpu]\" (or \".[roboflow-cpu]\")."
        ) from exc

    _log_onnx_runtime_provider_status()
    model = get_roboflow_model(model_id=roboflow_model_id, api_key=roboflow_api_key)
    return RoboflowPreprocessor(model=model, class_name=class_name, pad=pad, bg=bg)


def prediction_to_mask(prediction: dict[str, Any], height: int, width: int) -> np.ndarray:
    points = prediction.get("points") or []
    if not points:
        return np.zeros((height, width), dtype=bool)

    polygon = []
    for point in points:
        x = int(point.get("x", 0))
        y = int(point.get("y", 0))
        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))
        polygon.append((x, y))

    if len(polygon) < 3:
        return np.zeros((height, width), dtype=bool)

    mask_image = Image.new("L", (width, height), color=0)
    draw = ImageDraw.Draw(mask_image)
    draw.polygon(polygon, outline=1, fill=1)
    return np.array(mask_image, dtype=np.uint8).astype(bool)


def get_best_bin_mask(
    result: dict[str, Any],
    height: int,
    width: int,
    class_name: str,
) -> tuple[np.ndarray | None, dict[str, Any] | None]:
    predictions = result.get("predictions", []) or []
    if not predictions:
        return None, None

    filtered = [pred for pred in predictions if pred.get("class") == class_name] or predictions
    best = max(filtered, key=lambda pred: float(pred.get("confidence", 0.0)))
    mask = prediction_to_mask(best, height, width)
    metadata = {
        "confidence": float(best.get("confidence", 0.0)),
        "class": best.get("class"),
    }
    return mask, metadata


def mask_to_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    return int(x0), int(y0), int(x1) + 1, int(y1) + 1


def crop_mask_pad_square(
    pil_image: Image.Image,
    mask: np.ndarray | None,
    pad: int = 10,
    bg: int = 0,
) -> tuple[Image.Image, tuple[int, int, int, int]] | None:
    image = np.array(pil_image.convert("RGB"))
    height, width = image.shape[:2]

    if mask is None or mask.shape != (height, width):
        return None

    bbox = mask_to_bbox(mask)
    if bbox is None:
        return None

    x0, y0, x1, y1 = bbox
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(width, x1 + pad)
    y1 = min(height, y1 + pad)

    crop = image[y0:y1, x0:x1].copy()
    crop_mask = mask[y0:y1, x0:x1].astype(bool)
    crop[~crop_mask] = bg

    crop_height, crop_width = crop.shape[:2]
    side = max(crop_height, crop_width)
    square = np.full((side, side, 3), bg, dtype=np.uint8)

    offset_y = (side - crop_height) // 2
    offset_x = (side - crop_width) // 2
    square[offset_y : offset_y + crop_height, offset_x : offset_x + crop_width] = crop

    return Image.fromarray(square), (x0, y0, x1, y1)
