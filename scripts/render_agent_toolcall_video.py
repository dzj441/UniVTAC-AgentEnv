#!/usr/bin/env python3
"""Render a completed Codex rollout as observation + tool-call sharing video."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent_env.artifacts import encode_observation_toolcall_video  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Post-render one H.264 video with each public observation on the left "
            "and the agent tool call(s) based on it on the right."
        )
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Completed Codex run containing observations/ and codex_tool_calls.jsonl.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output MP4; defaults to <run_dir>/agent_timeline_h264.mp4.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    observation_root = run_dir / "observations"
    tool_calls = run_dir / "codex_tool_calls.jsonl"
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
    if not tool_calls.is_file():
        raise FileNotFoundError(f"Codex tool-call stream does not exist: {tool_calls}")
    output = (
        args.output.resolve()
        if args.output is not None
        else run_dir / "agent_timeline_h264.mp4"
    )
    metadata = encode_observation_toolcall_video(
        observation_root,
        tool_calls,
        output,
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
