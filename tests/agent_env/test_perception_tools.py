from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from agent_env.capabilities import CapabilityGateway, build_tool_registry
from agent_env.perception_runtime import (
    PerceptionExecution,
    PerceptionRuntime,
    PerceptionServiceError,
    _bbox,
)
from semantic_tools.http_server import serve
from tests.agent_env.test_capabilities import FakeSimulator, decision


SAM_REVISION = "3c879f39826c281e95690f02c7821c4de09afae7"
DEPTH_REVISION = "52b349b514bd8b47642f67ac78cb7b5dc5c51dd9"


@pytest.mark.parametrize(
    "bbox",
    ([-1, 0, 10, 10], [0, -1, 10, 10], [0, 0, 481, 10], [0, 0, 10, 271]),
)
def test_sam_bbox_must_stay_inside_public_rgb(bbox: list[int]) -> None:
    with pytest.raises(PerceptionServiceError, match="outside"):
        _bbox(bbox)


def test_semantic_service_refuses_non_loopback_bind() -> None:
    class _Service:
        route = "/test"

        def health(self) -> dict[str, Any]:
            return {"success": True}

        def handle(self, payload: dict[str, Any]) -> dict[str, Any]:
            return payload

    with pytest.raises(ValueError, match="loopback"):
        serve(_Service(), host="0.0.0.0", port=0)


def _png(width: int = 480, height: int = 270) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (40, 80, 120)).save(buffer, format="PNG")
    return buffer.getvalue()


def _npy(value: np.ndarray) -> str:
    buffer = io.BytesIO()
    np.save(buffer, value, allow_pickle=False)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def log_message(self, *_: object) -> None:
        return

    def _reply(self, value: dict[str, Any]) -> None:
        body = json.dumps(value).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        service = self.server.semantic_service  # type: ignore[attr-defined]
        model = "facebook/sam3" if service == "sam3" else "lpiccinelli/unidepth-v2-vitl14"
        revision = SAM_REVISION if service == "sam3" else DEPTH_REVISION
        self._reply(
            {
                "success": True,
                "service": service,
                "model": model,
                "revision": revision,
                "device": "cpu",
                "model_loaded": False,
            }
        )

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers["Content-Length"])
        request = json.loads(self.rfile.read(length))
        if self.path.endswith("segment"):
            image = base64.b64encode(_png()).decode("ascii")
            self._reply(
                {
                    "success": True,
                    "model": "facebook/sam3",
                    "revision": SAM_REVISION,
                    "duration_seconds": 0.1,
                    "image_size": [480, 270],
                    "detections": [
                        {
                            "rank": 0,
                            "label": request["prompt"],
                            "score": 0.9,
                            "bbox_xyxy": [1, 1, 479, 269],
                            "area_px": 128_104,
                            "mask_png_base64": image,
                            "overlay_png_base64": image,
                        }
                    ],
                }
            )
        else:
            depth = np.linspace(0.2, 1.2, 480 * 270, dtype=np.float32).reshape(270, 480)
            confidence = np.ones_like(depth) * 0.75
            self._reply(
                {
                    "success": True,
                    "model": "lpiccinelli/unidepth-v2-vitl14",
                    "revision": DEPTH_REVISION,
                    "resolution_level": request["resolution_level"],
                    "duration_seconds": 0.2,
                    "image_size": [480, 270],
                    "depth_npy_base64": _npy(depth),
                    "confidence_npy_base64": _npy(confidence),
                }
            )


def _server(service: str) -> tuple[ThreadingHTTPServer, threading.Thread]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    server.semantic_service = service  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def test_perception_runtime_materializes_outputs_and_bypasses_proxy(tmp_path: Path) -> None:
    sam, sam_thread = _server("sam3")
    depth, depth_thread = _server("unidepth_v2")
    source = tmp_path / "observations" / "obs_000" / "head_rgb.png"
    source.parent.mkdir(parents=True)
    source.write_bytes(_png())
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    old_proxy = os.environ.get("HTTP_PROXY")
    os.environ["HTTP_PROXY"] = "http://127.0.0.1:1"
    try:
        runtime = PerceptionRuntime(
            profile="sam3_unidepth_v2",
            run_dir=tmp_path,
            sam3_url=f"http://127.0.0.1:{sam.server_port}",
            unidepth_v2_url=f"http://127.0.0.1:{depth.server_port}",
        )
        assert set(runtime.health_manifest()["services"]) == {"sam3", "unidepth_v2"}
        segmented = runtime.segment_sam3(
            observation_id="obs_000",
            camera="head_rgb",
            image_path=source,
            image_sha256=digest,
            mode="text",
            prompt="dark prism",
            points=None,
            confidence_threshold=0.5,
        )
        estimated = runtime.estimate_unidepth_v2(
            observation_id="obs_000",
            camera="head_rgb",
            image_path=source,
            image_sha256=digest,
            intrinsics={"fx": 320.0, "fy": 320.0, "cx": 239.5, "cy": 134.5},
            resolution_level=4,
            sample_points=[{"x": 240.0, "y": 135.0}],
        )
        cached = runtime.estimate_unidepth_v2(
            observation_id="obs_000",
            camera="head_rgb",
            image_path=source,
            image_sha256=digest,
            intrinsics={"fx": 320.0, "fy": 320.0, "cx": 239.5, "cy": 134.5},
            resolution_level=4,
            sample_points=[],
        )
        assert segmented.success and segmented.public_response["detection_count"] == 1
        assert estimated.success and estimated.public_response["sample_count"] == 1
        assert cached.public_response["cache_hit"] is True
        assert all(path.is_file() for _, path, _ in (*segmented.content_images, *estimated.content_images))
    finally:
        if old_proxy is None:
            os.environ.pop("HTTP_PROXY", None)
        else:
            os.environ["HTTP_PROXY"] = old_proxy
        sam.shutdown()
        depth.shutdown()
        sam_thread.join(timeout=2)
        depth_thread.join(timeout=2)


class _FakePerceptionRuntime:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def segment_sam3(self, **kwargs: Any) -> PerceptionExecution:
        self.calls.append(kwargs)
        return PerceptionExecution(
            success=True,
            public_response={
                "status": "semantic_perception_complete",
                "tool": "sam3_segment",
                "observation_id": kwargs["observation_id"],
                "camera": kwargs["camera"],
                "detection_count": 1,
            },
            backend_request={"service": "sam3", "prompt": kwargs["prompt"]},
            content_images=(),
        )

    def estimate_unidepth_v2(self, **kwargs: Any) -> PerceptionExecution:
        self.calls.append(kwargs)
        return PerceptionExecution(
            success=True,
            public_response={
                "status": "semantic_perception_complete",
                "tool": "estimate_metric_depth",
                "observation_id": kwargs["observation_id"],
                "camera": kwargs["camera"],
                "depth_statistics": {"median_m": 0.5},
            },
            backend_request={
                "service": "unidepth_v2",
                "calibrated_intrinsics_supplied_by_host": True,
            },
            content_images=(),
        )


def test_gateway_perception_is_read_only_current_observation_and_evidence_gated(
    tmp_path: Path,
) -> None:
    simulator = FakeSimulator(tmp_path, 1)
    runtime = _FakePerceptionRuntime()
    gateway = CapabilityGateway(
        level=1,
        simulator_request=simulator,
        simulator_run_dir=tmp_path,
        perception_profile="sam3",
        perception_runtime=runtime,  # type: ignore[arg-type]
    )
    assert "sam3_segment" in build_tool_registry(1, "sam3")
    gateway.execute("start_episode", {"agent_note": "semantic test"})
    before = len(simulator.calls)
    premature = gateway.execute(
        "commit_classification",
        {
            "observation_id": "obs_000",
            "predicted_class": "rough",
            "target_pad": "orange",
            "decision_record": decision("sam3_result"),
        },
    )
    assert premature.success is False
    semantic = gateway.execute(
        "sam3_segment",
        {
            "observation_id": "obs_000",
            "camera": "head_rgb",
            "mode": "text",
            "prompt": "dark prism",
            "decision_record": decision(),
        },
    )
    assert semantic.success is True
    assert semantic.execution_target == "perception"
    assert len(simulator.calls) == before
    committed = gateway.execute(
        "commit_classification",
        {
            "observation_id": "obs_000",
            "predicted_class": "rough",
            "target_pad": "orange",
            "decision_record": decision("sam3_result"),
        },
    )
    assert committed.success is True
    assert len(simulator.calls) == before + 1


def test_gateway_uses_private_intrinsics_without_returning_them(tmp_path: Path) -> None:
    simulator = FakeSimulator(tmp_path, 1)
    runtime = _FakePerceptionRuntime()
    gateway = CapabilityGateway(
        level=1,
        simulator_request=simulator,
        simulator_run_dir=tmp_path,
        perception_profile="unidepth_v2",
        perception_runtime=runtime,  # type: ignore[arg-type]
    )
    gateway.execute("start_episode", {"agent_note": "calibration test"})
    sidecar = tmp_path / ".host_sensor_metadata" / "obs_000.json"
    sidecar.parent.mkdir(mode=0o700)
    sidecar.write_text(
        json.dumps(
            {
                "schema_version": "univtac.host_sensor_metadata.v1",
                "observation_id": "obs_000",
                "cameras": {
                    "head_rgb": {
                        "width": 480,
                        "height": 270,
                        "intrinsics": {
                            "fx": 320.0,
                            "fy": 321.0,
                            "cx": 239.5,
                            "cy": 134.5,
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    result = gateway.execute(
        "estimate_metric_depth",
        {
            "observation_id": "obs_000",
            "camera": "head_rgb",
            "resolution_level": 3,
            "sample_points": [{"x": 100, "y": 120}],
            "decision_record": decision(),
        },
    )
    assert result.success is True
    assert runtime.calls[-1]["intrinsics"] == {
        "fx": 320.0,
        "fy": 321.0,
        "cx": 239.5,
        "cy": 134.5,
    }
    rendered = json.dumps(result.public_response)
    assert "intrinsics" not in rendered
    assert "320.0" not in rendered


def test_gateway_enforces_per_observation_semantic_call_budget(tmp_path: Path) -> None:
    simulator = FakeSimulator(tmp_path, 1)
    runtime = _FakePerceptionRuntime()
    gateway = CapabilityGateway(
        level=1,
        simulator_request=simulator,
        simulator_run_dir=tmp_path,
        perception_profile="sam3",
        perception_runtime=runtime,  # type: ignore[arg-type]
    )
    gateway.execute("start_episode", {"agent_note": "budget test"})
    for index in range(4):
        result = gateway.execute(
            "sam3_segment",
            {
                "observation_id": "obs_000",
                "camera": "head_rgb",
                "mode": "text",
                "prompt": f"candidate {index}",
                "decision_record": decision(),
            },
        )
        assert result.success is True
        assert "call_index_for_observation" not in result.public_response
        assert "remaining_calls_for_observation" not in result.public_response
        assert result.raw_response is not None
        assert result.raw_response["call_index_for_observation"] == index + 1
        assert result.raw_response["remaining_calls_for_observation"] == 3 - index
    rejected = gateway.execute(
        "sam3_segment",
        {
            "observation_id": "obs_000",
            "camera": "head_rgb",
            "mode": "text",
            "prompt": "fifth candidate",
            "decision_record": decision(),
        },
    )
    assert rejected.success is False
    assert rejected.execution_target == "rejected"
    assert "budget exhausted" not in rejected.public_response["message"]
    assert rejected.raw_response is not None
    assert "budget exhausted" in rejected.raw_response["message"]
    assert len(runtime.calls) == 4
