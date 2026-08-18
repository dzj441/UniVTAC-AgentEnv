"""Media and audit-artifact helpers shared by AgentEnv simulator runners."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


TOOL_CALL_PANEL_WIDTH = 960
TOOL_CALL_PANEL_PADDING = 22
TOOL_CALL_HEADER_HEIGHT = 64
TOOL_CALL_CARD_GAP = 10
TOOL_CALL_FRAME_MIN_HEIGHT = 480


def tensor_to_rgb(value: Any) -> np.ndarray:
    array = value.detach().cpu().numpy()
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 3 or array.shape[-1] < 3:
        raise ValueError(f"Expected HWC RGB tensor, got shape {array.shape}")
    array = array[..., :3]
    if np.issubdtype(array.dtype, np.floating):
        finite_max = float(np.nanmax(array)) if array.size else 0.0
        if finite_max <= 1.0:
            array = array * 255.0
    return np.nan_to_num(array, nan=0.0, posinf=255.0, neginf=0.0).clip(0, 255).astype(np.uint8)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_rgb(path: Path, array: np.ndarray) -> dict[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, mode="RGB").save(path)
    return {"path": str(path.resolve()), "sha256": file_sha256(path)}


def marker_grid_health(array: np.ndarray) -> dict[str, Any]:
    """Reject the broken non-grid projection observed in early trials."""

    dark_mask = (array.astype(np.float32).mean(axis=2) < 45.0).astype(np.uint8)
    component_count, _, stats, _ = cv2.connectedComponentsWithStats(dark_mask, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    plausible = areas[(areas >= 25) & (areas <= 400)]
    return {
        "expected_markers": 63,
        "plausible_marker_components": int(len(plausible)),
        "all_dark_components": int(component_count - 1),
        "dark_pixel_count": int(dark_mask.sum()),
        "healthy": bool(len(plausible) >= 40),
    }


def require_initial_tactile_health(
    tactile_health: dict[str, dict[str, Any]],
    *,
    initial_observation: bool,
) -> None:
    """Fail only when a broken tactile renderer is present at reset.

    Strong contact can legitimately obscure or merge marker components. Once
    an episode has started, that is observation data for recovery, not a reason
    to discard an already executed action.
    """

    if initial_observation and not all(
        bool(sensor.get("healthy")) for sensor in tactile_health.values()
    ):
        raise RuntimeError(f"Initial tactile marker-grid health check failed: {tactile_health}")


def save_composite(
    panels: list[tuple[str, np.ndarray]],
    path: Path,
    *,
    columns: int = 2,
) -> dict[str, str]:
    if not panels:
        raise ValueError("At least one panel is required")
    panel_width, panel_height, label_height = 480, 270, 24
    rows = math.ceil(len(panels) / columns)
    canvas = Image.new(
        "RGB",
        (panel_width * columns, (panel_height + label_height) * rows),
        "black",
    )
    draw = ImageDraw.Draw(canvas)
    for index, (label, array) in enumerate(panels):
        column, row = index % columns, index // columns
        x = column * panel_width
        y = row * (panel_height + label_height)
        draw.text((x + 6, y + 5), label, fill="white")
        image = Image.fromarray(array, mode="RGB").resize(
            (panel_width, panel_height), Image.Resampling.BILINEAR
        )
        canvas.paste(image, (x, y + label_height))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    return {"path": str(path.resolve()), "sha256": file_sha256(path)}


def _media_executable(name: str) -> str:
    on_path = shutil.which(name)
    if on_path is not None:
        return on_path
    beside_python = Path(sys.executable).resolve().parent / name
    if beside_python.is_file():
        return str(beside_python)
    raise FileNotFoundError(f"Could not find {name!r} on PATH or beside {sys.executable}")


def _h264_encoder_args(ffmpeg: str) -> list[str]:
    encoders = subprocess.run(
        [ffmpeg, "-hide_banner", "-encoders"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if "libx264" in encoders:
        return ["-c:v", "libx264", "-profile:v", "main", "-crf", "23"]
    if "libopenh264" in encoders:
        return ["-c:v", "libopenh264", "-profile:v", "main", "-b:v", "2M"]
    raise RuntimeError("No H.264 encoder is available in the active ffmpeg build")


def _encode_h264(
    frame_pattern: str,
    video_path: Path,
    *,
    frames_per_second: float = 2.0,
) -> dict[str, Any]:
    ffmpeg = _media_executable("ffmpeg")
    ffprobe = _media_executable("ffprobe")
    video_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-framerate",
            str(frames_per_second),
            "-pattern_type",
            "glob",
            "-i",
            frame_pattern,
            *_h264_encoder_args(ffmpeg),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(video_path),
        ],
        check=True,
    )
    probe = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,profile,pix_fmt,width,height,nb_frames",
            "-of",
            "json",
            str(video_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    stream = json.loads(probe.stdout)["streams"][0]
    if stream.get("codec_name") != "h264" or stream.get("pix_fmt") != "yuv420p":
        raise RuntimeError(f"Unexpected replay video format: {stream}")
    return {
        "path": str(video_path.resolve()),
        "sha256": file_sha256(video_path),
        **stream,
    }


def encode_observation_video(observation_root: Path, video_path: Path) -> dict[str, Any]:
    """Encode the evaluator-owned, sensor-only public observation replay."""

    frame_pattern = str(observation_root / "obs_*" / "composite.png")
    return _encode_h264(frame_pattern, video_path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
        if isinstance(value, dict):
            records.append(value)
    return records


def tool_calls_by_observation(tool_call_path: Path) -> dict[str, list[dict[str, Any]]]:
    """Group attempts by the public observation that was current when called.

    ``prior_observation_id`` is assigned by the host gateway, so a malformed or
    stale observation id supplied by the agent still appears beside the frame
    the agent actually had.  ``start_episode`` has no prior observation and is
    intentionally omitted: it creates ``obs_000`` rather than acting on it.
    """

    grouped: dict[str, list[dict[str, Any]]] = {}
    for call in _read_jsonl(tool_call_path):
        observation_id = call.get("prior_observation_id")
        if not isinstance(observation_id, str) or not observation_id:
            continue
        grouped.setdefault(observation_id, []).append(call)
    for calls in grouped.values():
        calls.sort(key=lambda item: int(item.get("sequence", 0)))
    return grouped


def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    names = (
        (
            "DejaVuSansMono-Bold.ttf",
            "DejaVuSans-Bold.ttf",
            "OpenSans-SemiBold.ttf",
            "NVIDIASans_Bd.ttf",
        )
        if bold
        else (
            "DejaVuSansMono.ttf",
            "DejaVuSans.ttf",
            "NVIDIASans_Rg.ttf",
        )
    )
    python_version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    roots = (
        Path(sys.prefix)
        / "lib"
        / python_version
        / "site-packages"
        / "matplotlib"
        / "mpl-data"
        / "fonts"
        / "ttf",
        Path(sys.prefix)
        / "lib"
        / python_version
        / "site-packages"
        / "omni"
        / "resources"
        / "fonts",
        Path("/usr/share/fonts/truetype/dejavu"),
    )
    for root in roots:
        for name in names:
            path = root / name
            if path.is_file():
                return ImageFont.truetype(str(path), size=size)
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def _compact_tool_arguments(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {"arguments": value}
    return {
        str(key): child
        for key, child in value.items()
        if key not in {"decision_record", "agent_note"}
    }


def _wrapped_json_lines(value: Any, *, width: int = 68) -> list[str]:
    encoded = json.dumps(value, ensure_ascii=False, separators=(", ", ": "))
    return textwrap.wrap(
        encoded,
        width=width,
        replace_whitespace=False,
        drop_whitespace=True,
        break_long_words=True,
        break_on_hyphens=False,
    ) or ["{}"]


def _rationale_lines(call: dict[str, Any], *, width: int = 74) -> list[str]:
    arguments = call.get("arguments")
    decision = arguments.get("decision_record") if isinstance(arguments, dict) else None
    rationale = decision.get("rationale") if isinstance(decision, dict) else None
    if not isinstance(rationale, str) or not rationale.strip():
        return []
    return textwrap.wrap(
        " ".join(rationale.split()),
        width=width,
        break_long_words=True,
        break_on_hyphens=False,
    )


def _number(value: Any, *, precision: int = 4) -> str:
    if not isinstance(value, (int, float)):
        return str(value)
    return f"{float(value):+.{precision}f}"


def _vector(value: Any, *, precision: int = 3) -> str:
    if not isinstance(value, (list, tuple)):
        return str(value)
    return "[" + ", ".join(_number(item, precision=precision) for item in value) + "]"


def _tool_argument_lines(call: dict[str, Any]) -> list[str]:
    """Format high-value arguments for large, glanceable video typography."""

    tool = str(call.get("tool") or "unknown")
    arguments = call.get("arguments")
    if not isinstance(arguments, dict):
        return _wrapped_json_lines({"arguments": arguments})
    if tool == "probe_gripper":
        return [f"GRIPPER DELTA  {_number(arguments.get('delta_gripper'))} m"]
    if tool == "commit_classification":
        predicted = str(arguments.get("predicted_class") or "?").upper()
        pad = str(arguments.get("target_pad") or "?").upper()
        return [f"CLASS  {predicted}   ->   PAD  {pad}"]
    if tool == "act_delta_ee":
        return [
            f"POSITION DELTA  {_vector(arguments.get('delta_position'))} m",
            (
                f"RPY DELTA  {_vector(arguments.get('delta_rpy'))} rad"
                f"   |   GRIPPER  {_number(arguments.get('delta_gripper'))} m"
            ),
        ]
    if tool == "wait_physics":
        return [f"WAIT  {arguments.get('steps', '?')} PHYSICS STEPS"]
    if tool == "finish_episode":
        return ["FINALIZE THE CURRENT EPISODE STATE"]
    return _wrapped_json_lines(_compact_tool_arguments(arguments))


def _call_mapping_line(call: dict[str, Any], observation_id: str) -> str:
    arguments = call.get("arguments")
    requested = arguments.get("observation_id") if isinstance(arguments, dict) else None
    next_observation = call.get("next_observation_id")
    destination = next_observation if isinstance(next_observation, str) else "no new observation"
    if isinstance(requested, str) and requested != observation_id:
        return f"HOST {observation_id}  |  REQUESTED {requested}  ->  {destination}"
    return f"OBS  {observation_id}  ->  {destination}"


def _call_status(call: dict[str, Any]) -> tuple[str, str, str, str]:
    target = str(call.get("execution_target") or "simulator").upper()
    success = call.get("success") is True
    rejected = target == "REJECTED"
    if success:
        return f"OK | {target}", "#0f766e", "#2dd4bf", "#ecfeff"
    if rejected:
        return "HOST REJECTED", "#881337", "#fb7185", "#fff1f2"
    return f"FAILED | {target}", "#7f1d1d", "#f87171", "#fff1f2"


def _minimum_tool_panel_height(calls: list[dict[str, Any]]) -> int:
    if not calls:
        return TOOL_CALL_HEADER_HEIGHT + 129
    card_heights = []
    for call in calls:
        argument_count = len(_tool_argument_lines(call))
        rationale_count = min(3, len(_rationale_lines(call)))
        card_heights.append(48 + argument_count * 28 + 25 + 10 + rationale_count * 23)
    return (
        TOOL_CALL_HEADER_HEIGHT
        + 22
        + sum(card_heights)
        + TOOL_CALL_CARD_GAP * (len(calls) - 1)
    )


def render_observation_toolcall_frame(
    composite_path: Path,
    observation_id: str,
    calls: list[dict[str, Any]],
    *,
    frame_height: int | None = None,
) -> Image.Image:
    """Place the public observation and calls based on it in one video frame."""

    with Image.open(composite_path) as source:
        observation = source.convert("RGB")
    width = observation.width + TOOL_CALL_PANEL_WIDTH
    height = max(
        observation.height,
        TOOL_CALL_FRAME_MIN_HEIGHT,
        _minimum_tool_panel_height(calls),
        frame_height or 0,
    )
    if width % 2:
        width += 1
    if height % 2:
        height += 1
    canvas = Image.new("RGB", (width, height), "#070b12")
    canvas.paste(observation, (0, 0))
    panel_x = observation.width
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((panel_x, 0, width, height), fill="#0c1420")
    draw.line((panel_x, 0, panel_x, height), fill="#2dd4bf", width=2)

    panel_title_font = _font(28, bold=True)
    tool_font = _font(26, bold=True)
    status_font = _font(20, bold=True)
    argument_font = _font(22, bold=True)
    mapping_font = _font(18, bold=True)
    rationale_font = _font(18)
    x = panel_x + TOOL_CALL_PANEL_PADDING
    y = 11
    draw.text(
        (x, y),
        f"AGENT TOOL CALLS  |  {observation_id}",
        font=panel_title_font,
        fill="#f8fafc",
    )
    y += TOOL_CALL_HEADER_HEIGHT
    bottom = height - 11

    if not calls:
        draw.rounded_rectangle(
            (x, y, width - TOOL_CALL_PANEL_PADDING, min(bottom, y + 118)),
            radius=10,
            fill="#16202e",
            outline="#475569",
            width=2,
        )
        draw.text((x + 16, y + 14), "NO TOOL CALL RECORDED", font=tool_font, fill="#cbd5e1")
        y += 57
        for line in textwrap.wrap(
            "This observation was terminal, or the agent turn ended before another call.",
            width=76,
        ):
            draw.text((x + 16, y), line, font=rationale_font, fill="#94a3b8")
            y += 23
        return canvas

    cards: list[dict[str, Any]] = []
    for call in calls:
        argument_lines = _tool_argument_lines(call)
        rationale_lines = _rationale_lines(call)
        core_height = 48 + len(argument_lines) * 28 + 25 + 10
        cards.append(
            {
                "call": call,
                "argument_lines": argument_lines,
                "rationale_lines": rationale_lines,
                "core_height": core_height,
                "shown_rationale": 0,
            }
        )

    available_height = bottom - y - TOOL_CALL_CARD_GAP * (len(cards) - 1)
    core_total = sum(int(card["core_height"]) for card in cards)
    rationale_capacity = max(0, (available_height - core_total) // 23)
    while rationale_capacity:
        changed = False
        for card in cards:
            shown = int(card["shown_rationale"])
            lines = card["rationale_lines"]
            if shown < min(3, len(lines)) and rationale_capacity:
                card["shown_rationale"] = shown + 1
                rationale_capacity -= 1
                changed = True
        if not changed:
            break

    card_right = width - TOOL_CALL_PANEL_PADDING
    for card_index, card in enumerate(cards):
        call = card["call"]
        rationale_count = int(card["shown_rationale"])
        card_height = int(card["core_height"]) + rationale_count * 23
        card_bottom = min(bottom, y + card_height)
        status, header_fill, border, header_text = _call_status(call)
        draw.rounded_rectangle(
            (x, y, card_right, card_bottom),
            radius=10,
            fill="#121d2b",
            outline=border,
            width=2,
        )
        header_bottom = min(card_bottom, y + 48)
        draw.rounded_rectangle(
            (x, y, card_right, header_bottom),
            radius=9,
            fill=header_fill,
        )
        sequence = int(call.get("sequence", 0)) + 1
        tool = str(call.get("tool") or "unknown").upper()
        draw.text((x + 13, y + 7), f"A{sequence:02d}  {tool}", font=tool_font, fill=header_text)
        status_width = draw.textbbox((0, 0), status, font=status_font)[2]
        draw.text(
            (card_right - status_width - 13, y + 12),
            status,
            font=status_font,
            fill=header_text,
        )
        cursor_y = y + 56
        for line in card["argument_lines"]:
            draw.text((x + 14, cursor_y), line, font=argument_font, fill="#fde047")
            cursor_y += 28
        draw.text(
            (x + 14, cursor_y),
            _call_mapping_line(call, observation_id),
            font=mapping_font,
            fill="#7dd3fc" if call.get("success") is True else "#fda4af",
        )
        cursor_y += 25
        for rationale_index in range(rationale_count):
            prefix = "WHY  " if rationale_index == 0 else "     "
            rationale_line = card["rationale_lines"][rationale_index]
            if (
                rationale_index + 1 == rationale_count
                and rationale_count < len(card["rationale_lines"])
            ):
                rationale_line = rationale_line[:70].rstrip() + " ..."
            draw.text(
                (x + 14, cursor_y),
                prefix + rationale_line,
                font=rationale_font,
                fill="#c4b5fd",
            )
            cursor_y += 23
        y = card_bottom + (TOOL_CALL_CARD_GAP if card_index + 1 < len(cards) else 0)
    return canvas


def encode_observation_toolcall_video(
    observation_root: Path,
    tool_call_path: Path,
    video_path: Path,
) -> dict[str, Any]:
    """Encode one frame per observation with its subsequent agent tool attempts."""

    composite_paths = sorted(observation_root.glob("obs_*/composite.png"))
    if not composite_paths:
        raise FileNotFoundError(f"No composite observation frames under {observation_root}")
    grouped = tool_calls_by_observation(tool_call_path)
    video_path.parent.mkdir(parents=True, exist_ok=True)
    frame_height = TOOL_CALL_FRAME_MIN_HEIGHT
    for composite_path in composite_paths:
        with Image.open(composite_path) as source:
            frame_height = max(frame_height, source.height)
        frame_height = max(
            frame_height,
            _minimum_tool_panel_height(grouped.get(composite_path.parent.name, [])),
        )
    if frame_height % 2:
        frame_height += 1
    with tempfile.TemporaryDirectory(
        prefix=".toolcall-video-",
        dir=video_path.parent,
    ) as temporary:
        frame_root = Path(temporary)
        for index, composite_path in enumerate(composite_paths):
            observation_id = composite_path.parent.name
            frame = render_observation_toolcall_frame(
                composite_path,
                observation_id,
                grouped.get(observation_id, []),
                frame_height=frame_height,
            )
            frame.save(frame_root / f"frame_{index:06d}.png")
        temporary_video = frame_root / "timeline.mp4"
        metadata = _encode_h264(
            str(frame_root / "frame_*.png"),
            temporary_video,
            frames_per_second=1.0,
        )
        temporary_video.replace(video_path)

    visible_call_count = sum(len(grouped.get(path.parent.name, [])) for path in composite_paths)
    return {
        **metadata,
        "path": str(video_path.resolve()),
        "sha256": file_sha256(video_path),
        "layout": "public_observation_plus_agent_tool_calls",
        "frames_per_second": 1.0,
        "observation_frame_count": len(composite_paths),
        "mapped_tool_call_count": visible_call_count,
        "mapping_rule": "calls are attached to their host-recorded prior_observation_id",
        "source_tool_calls": str(tool_call_path.resolve()),
    }
