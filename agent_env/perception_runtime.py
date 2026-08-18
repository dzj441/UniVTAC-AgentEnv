"""Host-owned clients and artifact materialization for semantic perception."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import threading
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from .perception_profiles import PerceptionProfile, get_perception_profile


JsonDict = dict[str, Any]
MAX_HTTP_RESPONSE_BYTES = 192 * 1024 * 1024
MAX_DECODED_ARTIFACT_BYTES = 128 * 1024 * 1024
PUBLIC_RGB_SIZE = (480, 270)
MODEL_LOCK_PATH = Path(__file__).resolve().parents[1] / "semantic_tools" / "versions.json"


class PerceptionServiceError(RuntimeError):
    """A configured model service failed or violated its response contract."""


@dataclass(frozen=True)
class PerceptionExecution:
    success: bool
    public_response: JsonDict
    backend_request: JsonDict
    content_images: tuple[tuple[str, Path, str], ...]


@dataclass
class _DepthCacheEntry:
    depth: Any
    confidence: Any
    public_base: JsonDict
    content_images: tuple[tuple[str, Path, str], ...]


class _JsonHttpClient:
    def __init__(self, base_url: str, *, timeout_seconds: float) -> None:
        parsed = urllib.parse.urlparse(base_url)
        if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("Semantic model services must use a localhost http:// URL")
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = float(timeout_seconds)
        # Explicitly bypass all environment proxy variables.  Model requests never
        # leave the machine and must not depend on notebook/VPN proxy state.
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def get(self, route: str) -> JsonDict:
        request = urllib.request.Request(self.base_url + route, method="GET")
        return self._request(request)

    def post(self, route: str, payload: JsonDict) -> JsonDict:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        request = urllib.request.Request(
            self.base_url + route,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        return self._request(request)

    def _request(self, request: urllib.request.Request) -> JsonDict:
        try:
            with self._opener.open(request, timeout=self.timeout_seconds) as response:
                data = response.read(MAX_HTTP_RESPONSE_BYTES + 1)
        except urllib.error.HTTPError as exc:
            data = exc.read(MAX_HTTP_RESPONSE_BYTES + 1)
            if len(data) > MAX_HTTP_RESPONSE_BYTES:
                raise PerceptionServiceError(
                    "semantic service error response exceeds size limit"
                ) from exc
            try:
                payload = json.loads(data)
            except json.JSONDecodeError as decode_exc:
                raise PerceptionServiceError(
                    f"semantic service returned HTTP {exc.code} with invalid JSON"
                ) from decode_exc
            if not isinstance(payload, dict):
                raise PerceptionServiceError(
                    f"semantic service returned HTTP {exc.code} with a non-object"
                ) from exc
            return payload
        except (urllib.error.URLError, TimeoutError) as exc:
            raise PerceptionServiceError(
                f"semantic service unavailable: {type(exc).__name__}"
            ) from exc
        if len(data) > MAX_HTTP_RESPONSE_BYTES:
            raise PerceptionServiceError("semantic service response exceeds size limit")
        try:
            payload = json.loads(data)
        except json.JSONDecodeError as exc:
            raise PerceptionServiceError("semantic service returned invalid JSON") from exc
        if not isinstance(payload, dict):
            raise PerceptionServiceError("semantic service returned a non-object")
        return payload


class PerceptionRuntime:
    """Invoke only configured services on host-resolved current RGB artifacts."""

    def __init__(
        self,
        *,
        profile: str | PerceptionProfile,
        run_dir: Path,
        sam3_url: str = "",
        unidepth_v2_url: str = "",
        timeout_seconds: float = 300.0,
    ) -> None:
        self.profile = get_perception_profile(profile)
        self.run_dir = run_dir.resolve()
        self.output_root = self.run_dir / "semantic_perception"
        self._contracts = _model_contracts()
        self._sam3 = (
            _JsonHttpClient(sam3_url, timeout_seconds=timeout_seconds)
            if self.profile.expose_sam3
            else None
        )
        self._unidepth = (
            _JsonHttpClient(unidepth_v2_url, timeout_seconds=timeout_seconds)
            if self.profile.expose_unidepth_v2
            else None
        )
        if self.profile.expose_sam3 and not sam3_url:
            raise ValueError("SAM3 profile requires a service URL")
        if self.profile.expose_unidepth_v2 and not unidepth_v2_url:
            raise ValueError("UniDepth V2 profile requires a service URL")
        self._counter = 0
        self._lock = threading.Lock()
        self._depth_cache: dict[tuple[str, str, int, str], _DepthCacheEntry] = {}

    def health_manifest(self) -> JsonDict:
        services: JsonDict = {}
        if self._sam3 is not None:
            services["sam3"] = _health(
                self._sam3.get("/health"), "sam3", self._contracts["sam3"]
            )
        if self._unidepth is not None:
            services["unidepth_v2"] = _health(
                self._unidepth.get("/health"),
                "unidepth_v2",
                self._contracts["unidepth_v2"],
            )
        return {"profile": self.profile.to_manifest(), "services": services}

    def segment_sam3(
        self,
        *,
        observation_id: str,
        camera: str,
        image_path: Path,
        image_sha256: str,
        mode: str,
        prompt: str | None,
        points: list[JsonDict] | None,
        confidence_threshold: float,
    ) -> PerceptionExecution:
        if self._sam3 is None:
            raise PerceptionServiceError("SAM3 is not enabled")
        image_bytes = _verified_png(image_path, image_sha256)
        request: JsonDict = {
            "image_base64": base64.b64encode(image_bytes).decode("ascii"),
            "mode": mode,
        }
        safe_request: JsonDict = {
            "service": "sam3",
            "observation_id": observation_id,
            "camera": camera,
            "image_sha256": image_sha256,
            "mode": mode,
        }
        if mode == "text":
            request.update(
                {"prompt": prompt, "confidence_threshold": confidence_threshold}
            )
            safe_request.update(
                {"prompt": prompt, "confidence_threshold": confidence_threshold}
            )
        else:
            request["points"] = points
            safe_request["points"] = points
        raw = self._sam3.post("/v1/segment", request)
        if raw.get("success") is not True:
            return _failed_execution("sam3", observation_id, camera, safe_request, raw)
        with self._lock:
            output_dir = self._next_output_dir(observation_id, "sam3")
        return self._materialize_sam3(
            raw,
            output_dir=output_dir,
            observation_id=observation_id,
            camera=camera,
            safe_request=safe_request,
        )

    def estimate_unidepth_v2(
        self,
        *,
        observation_id: str,
        camera: str,
        image_path: Path,
        image_sha256: str,
        intrinsics: JsonDict,
        resolution_level: int,
        sample_points: list[JsonDict],
    ) -> PerceptionExecution:
        if self._unidepth is None:
            raise PerceptionServiceError("UniDepth V2 is not enabled")
        image_bytes = _verified_png(image_path, image_sha256)
        safe_request: JsonDict = {
            "service": "unidepth_v2",
            "observation_id": observation_id,
            "camera": camera,
            "image_sha256": image_sha256,
            "resolution_level": resolution_level,
            "sample_points": sample_points,
            "calibrated_intrinsics_supplied_by_host": True,
        }
        key = (observation_id, camera, resolution_level, image_sha256)
        with self._lock:
            cached = self._depth_cache.get(key)
        if cached is None:
            raw = self._unidepth.post(
                "/v1/estimate-depth",
                {
                    "image_base64": base64.b64encode(image_bytes).decode("ascii"),
                    "intrinsics": intrinsics,
                    "resolution_level": resolution_level,
                },
            )
            if raw.get("success") is not True:
                return _failed_execution(
                    "unidepth_v2", observation_id, camera, safe_request, raw
                )
            with self._lock:
                output_dir = self._next_output_dir(observation_id, "unidepth_v2")
            cached = self._materialize_depth(
                raw,
                output_dir=output_dir,
                observation_id=observation_id,
                camera=camera,
            )
            with self._lock:
                self._depth_cache[key] = cached
            cache_hit = False
        else:
            cache_hit = True
        samples = _sample_depth(cached.depth, cached.confidence, sample_points)
        public = {
            **cached.public_base,
            "cache_hit": cache_hit,
            "samples": samples,
            "sample_count": len(samples),
        }
        return PerceptionExecution(
            success=True,
            public_response=public,
            backend_request=safe_request,
            content_images=cached.content_images,
        )

    def _next_output_dir(self, observation_id: str, tool: str) -> Path:
        index = self._counter
        self._counter += 1
        path = self.output_root / observation_id / tool / f"call_{index:03d}"
        path.mkdir(parents=True, exist_ok=False)
        return path

    def _materialize_sam3(
        self,
        raw: JsonDict,
        *,
        output_dir: Path,
        observation_id: str,
        camera: str,
        safe_request: JsonDict,
    ) -> PerceptionExecution:
        _validate_model_identity(raw, self._contracts["sam3"])
        if raw.get("image_size") != list(PUBLIC_RGB_SIZE):
            raise PerceptionServiceError("SAM3 response dimensions do not match source RGB")
        detections = raw.get("detections")
        if not isinstance(detections, list) or len(detections) > 32:
            raise PerceptionServiceError("SAM3 returned an invalid detection list")
        public_detections: list[JsonDict] = []
        overlays: list[Path] = []
        for expected_rank, item in enumerate(detections):
            if not isinstance(item, dict):
                raise PerceptionServiceError("SAM3 returned an invalid detection")
            rank = _strict_int(item.get("rank"), "rank", 0, 31)
            if rank != expected_rank:
                raise PerceptionServiceError("SAM3 detection ranking is inconsistent")
            score = item.get("score")
            if score is not None:
                score = _finite_float(score, "score")
            bbox = _bbox(item.get("bbox_xyxy"))
            area = _strict_int(
                item.get("area_px"),
                "area_px",
                1,
                PUBLIC_RGB_SIZE[0] * PUBLIC_RGB_SIZE[1],
            )
            mask_path = output_dir / f"detection_{rank:03d}_mask.png"
            overlay_path = output_dir / f"detection_{rank:03d}_overlay.png"
            if _write_base64_png(mask_path, item.get("mask_png_base64")) != PUBLIC_RGB_SIZE:
                raise PerceptionServiceError("SAM3 mask dimensions do not match source RGB")
            if _write_base64_png(overlay_path, item.get("overlay_png_base64")) != PUBLIC_RGB_SIZE:
                raise PerceptionServiceError("SAM3 overlay dimensions do not match source RGB")
            overlays.append(overlay_path)
            public_detections.append(
                {
                    "detection_id": f"detection_{rank:03d}",
                    "rank": rank,
                    "label": str(item.get("label", ""))[:256],
                    "score": score,
                    "bbox_xyxy": bbox,
                    "area_px": area,
                    "mask": _artifact(mask_path, self.run_dir),
                    "overlay": _artifact(overlay_path, self.run_dir),
                }
            )
        contact_sheet = output_dir / "candidate_contact_sheet.png"
        _contact_sheet(overlays, contact_sheet)
        contact_artifact = _artifact(contact_sheet, self.run_dir)
        public = {
            "status": "semantic_perception_complete",
            "tool": "sam3_segment",
            "observation_id": observation_id,
            "camera": camera,
            "model": _required_response_text(raw, "model"),
            "model_revision": _required_response_text(raw, "revision"),
            "mode": safe_request["mode"],
            "detection_count": len(public_detections),
            "ranking": "score_descending",
            "detections": public_detections,
            "candidate_contact_sheet": contact_artifact,
            "model_duration_seconds": _finite_float(
                raw.get("duration_seconds"), "duration_seconds"
            ),
            "note": (
                "SAM3 scores rank mask quality/detections; they are not proof of "
                "task identity or calibrated probabilities."
            ),
        }
        images: tuple[tuple[str, Path, str], ...] = ()
        if contact_sheet.is_file():
            images = (("sam3_candidate_contact_sheet", contact_sheet, contact_artifact["sha256"]),)
        return PerceptionExecution(True, public, safe_request, images)

    def _materialize_depth(
        self,
        raw: JsonDict,
        *,
        output_dir: Path,
        observation_id: str,
        camera: str,
    ) -> _DepthCacheEntry:
        _validate_model_identity(raw, self._contracts["unidepth_v2"])
        depth = _decode_npy(raw.get("depth_npy_base64"), "depth")
        confidence = _decode_npy(raw.get("confidence_npy_base64"), "confidence")
        if raw.get("image_size") != list(PUBLIC_RGB_SIZE):
            raise PerceptionServiceError("UniDepth V2 response dimensions do not match source RGB")
        if depth.shape != confidence.shape or depth.ndim != 2:
            raise PerceptionServiceError("UniDepth V2 arrays have inconsistent shapes")
        if depth.shape != (PUBLIC_RGB_SIZE[1], PUBLIC_RGB_SIZE[0]):
            raise PerceptionServiceError("UniDepth V2 arrays do not match source RGB")
        valid = np.isfinite(depth) & (depth > 0)
        if not bool(valid.any()):
            raise PerceptionServiceError("UniDepth V2 returned no valid positive depth")
        depth_path = output_dir / "predicted_metric_depth.npy"
        confidence_path = output_dir / "relative_confidence.npy"
        np.save(depth_path, depth.astype(np.float32), allow_pickle=False)
        np.save(confidence_path, confidence.astype(np.float32), allow_pickle=False)
        visualization = output_dir / "depth_confidence_visualization.png"
        _depth_visualization(depth, confidence, visualization)
        values = depth[valid]
        stats = {
            "min_m": float(values.min()),
            "p05_m": float(np.percentile(values, 5)),
            "median_m": float(np.median(values)),
            "p95_m": float(np.percentile(values, 95)),
            "max_m": float(values.max()),
            "valid_ratio": float(valid.mean()),
        }
        vis_artifact = _artifact(visualization, self.run_dir)
        public = {
            "status": "semantic_perception_complete",
            "tool": "estimate_metric_depth",
            "observation_id": observation_id,
            "camera": camera,
            "model": _required_response_text(raw, "model"),
            "model_revision": _required_response_text(raw, "revision"),
            "resolution_level": _strict_int(raw.get("resolution_level"), "resolution_level", 0, 9),
            "depth_units": "metres",
            "depth_statistics": stats,
            "confidence_semantics": "relative_within_image_higher_is_better",
            "predicted_depth": _artifact(depth_path, self.run_dir),
            "relative_confidence": _artifact(confidence_path, self.run_dir),
            "visualization": vis_artifact,
            "model_duration_seconds": _finite_float(
                raw.get("duration_seconds"), "duration_seconds"
            ),
            "note": (
                "This is a monocular model prediction, not simulator/sensor ground-truth "
                "depth and not final collision-clearance evidence."
            ),
        }
        return _DepthCacheEntry(
            depth=depth,
            confidence=confidence,
            public_base=public,
            content_images=(("unidepth_v2_visualization", visualization, vis_artifact["sha256"]),),
        )


def _model_contracts() -> dict[str, JsonDict]:
    try:
        lock = json.loads(MODEL_LOCK_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PerceptionServiceError("semantic model lock is unavailable") from exc
    contracts: dict[str, JsonDict] = {}
    for name in ("sam3", "unidepth_v2"):
        item = lock.get(name) if isinstance(lock, dict) else None
        if not isinstance(item, dict):
            raise PerceptionServiceError("semantic model lock is malformed")
        model, revision = item.get("model_id"), item.get("model_revision")
        if not isinstance(model, str) or not isinstance(revision, str):
            raise PerceptionServiceError("semantic model lock is malformed")
        contracts[name] = {"model": model, "revision": revision}
    return contracts


def _validate_model_identity(payload: JsonDict, contract: JsonDict) -> None:
    if (
        _required_response_text(payload, "model") != contract["model"]
        or _required_response_text(payload, "revision") != contract["revision"]
    ):
        raise PerceptionServiceError("semantic service model identity violates the lock")


def _health(payload: JsonDict, service: str, contract: JsonDict) -> JsonDict:
    if payload.get("success") is not True or payload.get("service") != service:
        raise PerceptionServiceError(f"{service} health check failed")
    _validate_model_identity(payload, contract)
    return {
        "service": service,
        "model": _required_response_text(payload, "model"),
        "revision": _required_response_text(payload, "revision"),
        "device": _required_response_text(payload, "device"),
        "model_loaded": bool(payload.get("model_loaded")),
    }


def _failed_execution(
    service: str,
    observation_id: str,
    camera: str,
    safe_request: JsonDict,
    raw: JsonDict,
) -> PerceptionExecution:
    response = {
        "status": "semantic_perception_error",
        "tool": service,
        "observation_id": observation_id,
        "camera": camera,
        "error": str(raw.get("error", "service_failure"))[:160],
        "error_type": str(raw.get("error_type", ""))[:160],
    }
    return PerceptionExecution(False, response, safe_request, ())


def _verified_png(path: Path, expected_sha256: str) -> bytes:
    data = path.read_bytes()
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected_sha256:
        raise PerceptionServiceError("source RGB hash does not match the observation artifact")
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise PerceptionServiceError("source RGB artifact is not a PNG")
    try:
        image = Image.open(io.BytesIO(data))
        size = image.size
        image.verify()
    except Exception as exc:  # noqa: BLE001
        raise PerceptionServiceError("source RGB artifact is an invalid PNG") from exc
    if size != PUBLIC_RGB_SIZE:
        raise PerceptionServiceError("source RGB dimensions violate the benchmark contract")
    return data


def _write_base64_png(path: Path, value: Any) -> tuple[int, int]:
    data = _decode_base64(value)
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise PerceptionServiceError("semantic service artifact is not a PNG")
    image = Image.open(io.BytesIO(data))
    size = image.size
    image.verify()
    path.write_bytes(data)
    return size


def _decode_npy(value: Any, name: str) -> Any:
    data = _decode_base64(value)
    try:
        array = np.load(io.BytesIO(data), allow_pickle=False)
    except Exception as exc:  # noqa: BLE001
        raise PerceptionServiceError(f"invalid {name} NPY artifact") from exc
    if not isinstance(array, np.ndarray) or array.dtype.kind not in "fiu":
        raise PerceptionServiceError(f"invalid {name} array")
    return np.asarray(array, dtype=np.float32)


def _decode_base64(value: Any) -> bytes:
    if not isinstance(value, str) or not value:
        raise PerceptionServiceError("semantic service omitted an artifact payload")
    try:
        data = base64.b64decode(value, validate=True)
    except ValueError as exc:
        raise PerceptionServiceError("semantic service returned invalid base64") from exc
    if len(data) > MAX_DECODED_ARTIFACT_BYTES:
        raise PerceptionServiceError("semantic service artifact exceeds size limit")
    return data


def _artifact(path: Path, run_dir: Path) -> JsonDict:
    resolved = path.resolve()
    if not resolved.is_relative_to(run_dir):
        raise PerceptionServiceError("perception artifact escaped the run directory")
    data = resolved.read_bytes()
    return {
        "artifact_id": str(resolved.relative_to(run_dir)),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _contact_sheet(overlays: list[Path], output: Path) -> None:
    if not overlays:
        image = Image.new("RGB", (480, 96), "#111827")
        ImageDraw.Draw(image).text((20, 36), "SAM3: no detections", fill="white")
        image.save(output, format="PNG")
        return
    images = [Image.open(path).convert("RGB") for path in overlays[:8]]
    width = max(image.width for image in images)
    height = max(image.height for image in images)
    columns = 2 if len(images) > 1 else 1
    rows = math.ceil(len(images) / columns)
    sheet = Image.new("RGB", (width * columns, height * rows), "black")
    for index, image in enumerate(images):
        sheet.paste(image, ((index % columns) * width, (index // columns) * height))
    sheet.save(output, format="PNG")


def _depth_visualization(depth: Any, confidence: Any, output: Path) -> None:
    valid = np.isfinite(depth) & (depth > 0)
    low, high = np.percentile(depth[valid], [2, 98])
    if not high > low:
        high = low + 1e-6
    normalized = np.clip((depth - low) / (high - low), 0.0, 1.0)
    normalized[~valid] = 0.0
    # Compact perceptual-ish blue -> cyan -> yellow -> red ramp.
    red = np.clip(1.5 - np.abs(4.0 * normalized - 3.0), 0.0, 1.0)
    green = np.clip(1.5 - np.abs(4.0 * normalized - 2.0), 0.0, 1.0)
    blue = np.clip(1.5 - np.abs(4.0 * normalized - 1.0), 0.0, 1.0)
    depth_rgb = (np.stack((red, green, blue), axis=-1) * 255).astype(np.uint8)
    finite_conf = confidence[np.isfinite(confidence)]
    if finite_conf.size:
        c_low, c_high = np.percentile(finite_conf, [2, 98])
        scale = max(float(c_high - c_low), 1e-6)
        conf = np.clip((confidence - c_low) / scale, 0.0, 1.0)
    else:
        conf = np.zeros_like(confidence)
    conf = np.nan_to_num(conf, nan=0.0)
    conf_rgb = np.repeat((conf[..., None] * 255).astype(np.uint8), 3, axis=-1)
    h, w = depth.shape
    canvas = Image.new("RGB", (w * 2, h + 30), "#111827")
    canvas.paste(Image.fromarray(depth_rgb, mode="RGB"), (0, 30))
    canvas.paste(Image.fromarray(conf_rgb, mode="RGB"), (w, 30))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), f"predicted depth {low:.3f}..{high:.3f} m", fill="white")
    draw.text((w + 8, 8), "relative confidence", fill="white")
    canvas.save(output, format="PNG")


def _sample_depth(depth: Any, confidence: Any, points: list[JsonDict]) -> list[JsonDict]:
    height, width = depth.shape
    samples = []
    for index, point in enumerate(points):
        x = _finite_float(point.get("x"), "sample x")
        y = _finite_float(point.get("y"), "sample y")
        if not 0 <= x < width or not 0 <= y < height:
            raise PerceptionServiceError("depth sample point lies outside the source image")
        px, py = int(round(x)), int(round(y))
        px, py = min(px, width - 1), min(py, height - 1)
        region = depth[max(0, py - 1) : min(height, py + 2), max(0, px - 1) : min(width, px + 2)]
        valid = region[np.isfinite(region) & (region > 0)]
        value = float(np.median(valid)) if valid.size else None
        conf = confidence[py, px]
        samples.append(
            {
                "sample_id": f"sample_{index:03d}",
                "x": x,
                "y": y,
                "depth_m_3x3_median": value,
                "relative_confidence": float(conf) if np.isfinite(conf) else None,
            }
        )
    return samples


def _bbox(value: Any) -> list[int]:
    if not isinstance(value, list) or len(value) != 4:
        raise PerceptionServiceError("SAM3 bbox must contain four values")
    result = []
    for item in value:
        number = _finite_float(item, "bbox")
        result.append(int(round(number)))
    if result[2] <= result[0] or result[3] <= result[1]:
        raise PerceptionServiceError("SAM3 bbox is empty")
    width, height = PUBLIC_RGB_SIZE
    if result[0] < 0 or result[1] < 0 or result[2] > width or result[3] > height:
        raise PerceptionServiceError("SAM3 bbox lies outside the source image")
    return result


def _required_response_text(value: JsonDict, key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item.strip():
        raise PerceptionServiceError(f"semantic response omitted {key}")
    return item.strip()[:256]


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise PerceptionServiceError(f"{name} must be finite")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise PerceptionServiceError(f"{name} must be finite") from exc
    if not math.isfinite(number):
        raise PerceptionServiceError(f"{name} must be finite")
    return number


def _strict_int(value: Any, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise PerceptionServiceError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value
