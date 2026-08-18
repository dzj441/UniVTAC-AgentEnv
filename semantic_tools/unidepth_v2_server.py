"""Pinned UniDepth V2 metric-depth-prior service for UniVTAC.

The inference contract follows OpenETA's UniDepth V2 service at commit
fbf102dfd30e44b981d50f1c29dede4274eaaf8f.  Calibrated intrinsics are supplied
by the host but are deliberately omitted from the agent-visible response.
"""

from __future__ import annotations

import argparse
import base64
import io
import math
import threading
import time
from pathlib import Path
from typing import Any

from .http_server import serve


MODEL_ID = "lpiccinelli/unidepth-v2-vitl14"
ROUTE = "/v1/estimate-depth"


class UniDepthV2Service:
    route = ROUTE

    def __init__(self, *, model_dir: Path, revision: str, device: str) -> None:
        self.model_dir = model_dir.resolve()
        self.revision = str(revision)
        self.device_request = str(device)
        if not self.model_dir.is_dir():
            raise FileNotFoundError(f"UniDepth V2 model directory not found: {self.model_dir}")
        self._model: Any | None = None
        self._torch: Any | None = None
        self._device = ""
        self._load_lock = threading.Lock()
        self._inference_lock = threading.Lock()

    def health(self) -> dict[str, Any]:
        return {
            "success": True,
            "service": "unidepth_v2",
            "schema_version": "univtac.unidepth_v2_service.v1",
            "model": MODEL_ID,
            "revision": self.revision,
            "device": self.device_request,
            "model_loaded": self._model is not None,
        }

    def handle(self, payload: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        np, image = _decode_image(payload.get("image_base64"))
        intrinsics = _intrinsics(payload.get("intrinsics"))
        level = _resolution_level(payload.get("resolution_level", 4))
        model, torch, device = self._get_model()
        rgb = torch.from_numpy(np.asarray(image, dtype=np.uint8).copy()).permute(2, 0, 1)
        rgb = rgb.contiguous().to(device)
        camera_matrix = torch.tensor(
            [
                [intrinsics["fx"], 0.0, intrinsics["cx"]],
                [0.0, intrinsics["fy"], intrinsics["cy"]],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
            device=device,
        )
        with self._inference_lock:
            model.resolution_level = level
            with torch.inference_mode():
                predictions = model.infer(rgb, camera_matrix)
        depth = _prediction(predictions, "depth", image.size[::-1], np)
        confidence = _prediction(predictions, "confidence", image.size[::-1], np)
        valid = np.isfinite(depth) & (depth > 0)
        if not bool(valid.any()):
            raise RuntimeError("UniDepth V2 returned no finite positive depth")
        return {
            "success": True,
            "schema_version": "univtac.unidepth_v2_service.v1",
            "model": MODEL_ID,
            "revision": self.revision,
            "device": device,
            "resolution_level": level,
            "image_size": [image.width, image.height],
            "depth_units": "metres",
            "confidence_semantics": "relative_within_image_higher_is_better",
            "valid_depth_ratio": float(valid.mean()),
            "depth_npy_base64": _encode_npy(depth, np),
            "confidence_npy_base64": _encode_npy(confidence, np),
            "duration_seconds": round(time.perf_counter() - started, 6),
        }

    def _get_model(self) -> tuple[Any, Any, str]:
        if self._model is not None and self._torch is not None:
            return self._model, self._torch, self._device
        with self._load_lock:
            if self._model is not None and self._torch is not None:
                return self._model, self._torch, self._device
            import torch
            from unidepth.models import UniDepthV2

            device = self.device_request
            if device == "auto":
                device = "cuda:0" if torch.cuda.is_available() else "cpu"
            if device.startswith("cuda") and not torch.cuda.is_available():
                raise RuntimeError("requested CUDA device is unavailable")
            model = UniDepthV2.from_pretrained(str(self.model_dir))
            self._model = model.to(device).eval()
            self._torch = torch
            self._device = device
            return self._model, self._torch, self._device


def _decode_image(encoded: Any) -> tuple[Any, Any]:
    import numpy as np
    from PIL import Image

    if not isinstance(encoded, str) or not encoded:
        raise ValueError("image_base64 must be non-empty")
    try:
        data = base64.b64decode(encoded, validate=True)
        image = Image.open(io.BytesIO(data)).convert("RGB")
        image.load()
    except Exception as exc:  # noqa: BLE001
        raise ValueError("image_base64 is not a valid PNG/JPEG") from exc
    if image.width < 2 or image.height < 2 or image.width * image.height > 32_000_000:
        raise ValueError("image dimensions are outside service limits")
    return np, image


def _intrinsics(value: Any) -> dict[str, float]:
    if not isinstance(value, dict) or set(value) != {"fx", "fy", "cx", "cy"}:
        raise ValueError("intrinsics must contain exactly fx, fy, cx, and cy")
    result = {}
    for key in ("fx", "fy", "cx", "cy"):
        if isinstance(value[key], bool):
            raise ValueError("intrinsics must be numeric")
        number = float(value[key])
        if not math.isfinite(number) or (key in {"fx", "fy"} and number <= 0):
            raise ValueError("intrinsics contain an invalid value")
        result[key] = number
    return result


def _resolution_level(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 10:
        raise ValueError("resolution_level must be an integer in [0, 9]")
    return value


def _prediction(predictions: Any, key: str, shape: tuple[int, int], np: Any) -> Any:
    if not isinstance(predictions, dict) or key not in predictions:
        raise RuntimeError(f"UniDepth V2 response is missing {key}")
    value = predictions[key]
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    array = np.asarray(value, dtype=np.float32).squeeze()
    if array.shape != tuple(shape):
        raise RuntimeError(
            f"UniDepth V2 {key} shape {array.shape} does not match image {tuple(shape)}"
        )
    return np.ascontiguousarray(array, dtype=np.float32)


def _encode_npy(array: Any, np: Any) -> str:
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UniVTAC UniDepth V2 localhost service")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8784)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    service = UniDepthV2Service(
        model_dir=args.model_dir,
        revision=args.revision,
        device=args.device,
    )
    serve(service, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
