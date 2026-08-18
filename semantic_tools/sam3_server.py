"""Pinned SAM3 text/point segmentation service for UniVTAC.

The model API and output normalization follow the OpenETA SAM3 service design
at commit fbf102dfd30e44b981d50f1c29dede4274eaaf8f.  This implementation uses a
smaller localhost JSON transport because Codex sees host-owned dynamic tools,
not the model service itself.
"""

from __future__ import annotations

import argparse
import base64
import io
import math
import threading
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from .http_server import serve


MODEL_ID = "facebook/sam3"
ROUTE = "/v1/segment"
MAX_POINTS = 64
MAX_DETECTIONS = 32


class Sam3Service:
    route = ROUTE

    def __init__(self, *, checkpoint: Path, revision: str, device: str) -> None:
        self.checkpoint = checkpoint.resolve()
        self.revision = str(revision)
        self.device = str(device)
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"SAM3 checkpoint not found: {self.checkpoint}")
        self._model: Any | None = None
        self._processor: Any | None = None
        self._lock = threading.Lock()

    def health(self) -> dict[str, Any]:
        return {
            "success": True,
            "service": "sam3",
            "schema_version": "univtac.sam3_service.v1",
            "model": MODEL_ID,
            "revision": self.revision,
            "device": self.device,
            "model_loaded": self._model is not None,
        }

    def handle(self, payload: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        image = _decode_image(payload.get("image_base64"))
        mode = str(payload.get("mode", "text")).strip().lower()
        if mode == "text":
            prompt = _required_text(payload.get("prompt"), "prompt", maximum=256)
            threshold = _threshold(payload.get("confidence_threshold", 0.5))
            detections = self._segment_text(image, prompt, threshold)
            request_descriptor: dict[str, Any] = {
                "mode": mode,
                "prompt": prompt,
                "confidence_threshold": threshold,
            }
        elif mode == "points":
            points = _validate_points(payload.get("points"), image.size)
            detections = self._segment_points(image, points)
            request_descriptor = {"mode": mode, "points": points}
        else:
            raise ValueError("mode must be exactly 'text' or 'points'")

        rendered = [_render_detection(image, item, request_descriptor) for item in detections]
        return {
            "success": True,
            "schema_version": "univtac.sam3_service.v1",
            "model": MODEL_ID,
            "revision": self.revision,
            **request_descriptor,
            "image_size": [image.width, image.height],
            "ranking": "score_descending",
            "detection_count": len(rendered),
            "detections": rendered,
            "duration_seconds": round(time.perf_counter() - started, 6),
        }

    def _get_processor(self, threshold: float) -> tuple[Any, Any]:
        import torch

        if self._processor is None:
            from sam3.model.sam3_image_processor import Sam3Processor
            from sam3.model_builder import build_sam3_image_model

            # The pinned upstream helper only moves the model when device is
            # exactly ``"cuda"``; passing the otherwise-valid ``"cuda:0"``
            # silently leaves weights on CPU.  Load deterministically on CPU,
            # then perform the explicit move ourselves.
            model = build_sam3_image_model(
                device="cpu",
                checkpoint_path=str(self.checkpoint),
                load_from_HF=False,
                enable_inst_interactivity=True,
            )
            self._model = model.to(self.device).eval()
            self._processor = Sam3Processor(
                self._model,
                device=self.device,
                confidence_threshold=threshold,
            )
        else:
            self._processor.set_confidence_threshold(threshold)
        return self._processor, torch

    def _segment_text(self, image: Any, prompt: str, threshold: float) -> list[dict[str, Any]]:
        import numpy as np

        with self._lock:
            processor, torch = self._get_processor(threshold)
            context = _autocast(torch, self.device)
            with context, torch.inference_mode():
                state = processor.set_image(image)
                output = processor.set_text_prompt(state=state, prompt=prompt)
            _sync(torch, self.device)
            masks = _normalise_masks(output.get("masks"), torch=torch, np=np)
            boxes = _normalise_boxes(output.get("boxes"), torch=torch, np=np)
            scores = _normalise_scores(output.get("scores"), torch=torch, np=np)
            del state, output
            _empty_cache(torch, self.device)
        if len(masks) != len(boxes) or len(scores) not in {0, len(masks)}:
            raise RuntimeError("SAM3 returned inconsistent text detection arrays")
        rows = []
        for index, mask in enumerate(masks[:MAX_DETECTIONS]):
            if not bool(mask.any()):
                continue
            rows.append(
                {
                    "backend_index": index,
                    "label": prompt,
                    "score": scores[index] if scores else None,
                    "bbox_xyxy": [int(round(value)) for value in boxes[index]],
                    "area_px": int(mask.sum()),
                    "mask_array": mask,
                }
            )
        return _rank(rows)

    def _segment_points(self, image: Any, points: list[dict[str, Any]]) -> list[dict[str, Any]]:
        import numpy as np

        with self._lock:
            processor, torch = self._get_processor(0.5)
            coords = np.asarray([[p["x"], p["y"]] for p in points], dtype=np.float32)
            labels = np.asarray([p["label"] for p in points], dtype=np.int32)
            context = _autocast(torch, self.device)
            with context, torch.inference_mode():
                state = processor.set_image(image)
                masks, scores, _logits = processor.model.predict_inst(
                    state,
                    point_coords=coords,
                    point_labels=labels,
                    multimask_output=True,
                )
            _sync(torch, self.device)
            masks = _normalise_masks(masks, torch=torch, np=np)
            scores = _normalise_scores(scores, torch=torch, np=np)
            predictor = getattr(processor.model, "inst_interactive_predictor", None)
            if predictor is not None:
                predictor._features = None
                predictor._is_image_set = False
            del state, _logits
            _empty_cache(torch, self.device)
        if len(masks) != 3 or len(scores) != 3:
            raise RuntimeError("SAM3 point mode must return exactly three candidates")
        rows = []
        for index, mask in enumerate(masks):
            if not bool(mask.any()):
                continue
            rows.append(
                {
                    "backend_index": index,
                    "label": "point_prompt",
                    "score": scores[index],
                    "bbox_xyxy": _mask_bbox(mask, np),
                    "area_px": int(mask.sum()),
                    "mask_array": mask,
                }
            )
        return _rank(rows)


def _decode_image(encoded: Any) -> Any:
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
    return image


def _required_text(value: Any, name: str, *, maximum: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty")
    result = value.strip()
    if len(result) > maximum:
        raise ValueError(f"{name} exceeds {maximum} characters")
    return result


def _threshold(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("confidence_threshold must be numeric")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError("confidence_threshold must be in [0, 1]")
    return number


def _validate_points(value: Any, image_size: tuple[int, int]) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not 1 <= len(value) <= MAX_POINTS:
        raise ValueError(f"points must contain 1..{MAX_POINTS} items")
    width, height = image_size
    points = []
    for item in value:
        if not isinstance(item, dict) or set(item) != {"x", "y", "label"}:
            raise ValueError("each point must contain exactly x, y, and label")
        if isinstance(item["x"], bool) or isinstance(item["y"], bool):
            raise ValueError("point coordinates must be numeric")
        x, y = float(item["x"]), float(item["y"])
        label = item["label"]
        if not math.isfinite(x) or not math.isfinite(y) or not 0 <= x < width or not 0 <= y < height:
            raise ValueError("point lies outside the source image")
        if isinstance(label, bool) or label not in (0, 1):
            raise ValueError("point label must be integer 0 or 1")
        points.append({"x": x, "y": y, "label": int(label)})
    if not any(point["label"] == 1 for point in points):
        raise ValueError("at least one foreground point is required")
    return points


def _autocast(torch: Any, device: str) -> Any:
    if device.startswith("cuda") and torch.cuda.is_available():
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _sync(torch: Any, device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _empty_cache(torch: Any, device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _normalise_masks(value: Any, *, torch: Any, np: Any) -> Any:
    if value is None:
        return np.zeros((0, 0, 0), dtype=bool)
    array = (
        value.detach().float().cpu().numpy()
        if isinstance(value, torch.Tensor)
        else np.asarray(value)
    )
    array = np.squeeze(array)
    if array.ndim == 2:
        array = array[None, ...]
    if array.ndim != 3:
        raise RuntimeError("SAM3 mask tensor has an unexpected shape")
    return array > 0


def _normalise_boxes(value: Any, *, torch: Any, np: Any) -> list[list[float]]:
    if value is None:
        return []
    array = (
        value.detach().float().cpu().numpy()
        if isinstance(value, torch.Tensor)
        else np.asarray(value)
    )
    array = np.squeeze(array)
    if array.size == 0:
        return []
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim != 2 or array.shape[1] != 4:
        raise RuntimeError("SAM3 boxes have an unexpected shape")
    return [[float(item) for item in row] for row in array.tolist()]


def _normalise_scores(value: Any, *, torch: Any, np: Any) -> list[float]:
    if value is None:
        return []
    array = (
        value.detach().float().cpu().numpy()
        if isinstance(value, torch.Tensor)
        else np.asarray(value)
    )
    return [float(item) for item in array.reshape(-1).tolist()]


def _mask_bbox(mask: Any, np: Any) -> list[int]:
    ys, xs = np.nonzero(mask)
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def _rank(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows.sort(
        key=lambda row: (
            row["score"] is not None and math.isfinite(float(row["score"])),
            float(row["score"]) if row["score"] is not None else -math.inf,
        ),
        reverse=True,
    )
    for rank, row in enumerate(rows):
        row["rank"] = rank
    return rows


def _render_detection(image: Any, row: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
    import numpy as np
    from PIL import Image, ImageDraw

    mask = row.pop("mask_array")
    mask_image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    overlay = np.asarray(image.convert("RGBA")).copy()
    color = np.asarray([0, 150, 255, 120], dtype=np.uint8)
    overlay[mask] = (
        0.55 * overlay[mask].astype(np.float32) + 0.45 * color.astype(np.float32)
    ).astype(np.uint8)
    rendered = Image.fromarray(overlay, mode="RGBA")
    draw = ImageDraw.Draw(rendered)
    if request["mode"] == "points":
        radius = max(4, round(min(image.size) / 80))
        for point in request["points"]:
            x, y = round(point["x"]), round(point["y"])
            colour = "#7CFC00" if point["label"] == 1 else "#FF3030"
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=colour, width=2)
    label = f"rank={row['rank']} score={row['score']} area={row['area_px']}"
    draw.rectangle((4, 4, min(image.width - 4, 340), 28), fill=(0, 0, 0, 190))
    draw.text((8, 8), label, fill="white")
    return {
        **row,
        "mask_png_base64": _encode_png(mask_image),
        "overlay_png_base64": _encode_png(rendered),
    }


def _encode_png(image: Any) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UniVTAC SAM3 localhost service")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8783)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    service = Sam3Service(
        checkpoint=args.checkpoint,
        revision=args.revision,
        device=args.device,
    )
    serve(service, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
