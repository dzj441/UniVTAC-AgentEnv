#!/usr/bin/env python3
"""Validate and freeze one replay-proven fixed expert trajectory manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent_env.expert_tasks import get_expert_task  # noqa: E402
from agent_env.expert_trajectory import build_fixed_expert_manifest  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze one successful full-episode expert trajectory"
    )
    parser.add_argument("--task", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--source-hdf5", required=True, type=Path)
    parser.add_argument("--source-video", required=True, type=Path)
    parser.add_argument("--replay-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    task = get_expert_task(args.task)
    manifest = build_fixed_expert_manifest(
        task=task.name,
        seed=args.seed,
        source_hdf5=args.source_hdf5,
        source_video=args.source_video,
        replay_report_path=args.replay_report,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
