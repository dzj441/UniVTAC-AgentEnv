#!/usr/bin/env python3
"""Export one authenticated fixed expert demonstration for an Agent workspace."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent_env.benchmark_profiles import (  # noqa: E402
    AnnotationCapabilities,
    get_observation_profile,
)
from agent_env.benchmark_tasks import list_benchmark_tasks  # noqa: E402
from agent_env.fixed_demo_bundle import (  # noqa: E402
    project_fixed_demo_bundle,
    resolve_fixed_demo_root,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Project a P6 expert master into a profile-safe ICL bundle"
    )
    parser.add_argument(
        "--task",
        required=True,
        choices=tuple(task.name for task in list_benchmark_tasks()),
    )
    parser.add_argument(
        "--profile", required=True, choices=tuple(str(index) for index in range(1, 7))
    )
    parser.add_argument("--provide-bbox", action="store_true")
    parser.add_argument("--provide-mask", action="store_true")
    parser.add_argument(
        "--fixed-demo-root",
        type=Path,
        help=(
            "Host-side directory containing registered P6 masters; defaults to "
            "UNIVTAC_FIXED_EXPERT_MASTER_ROOT"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New Agent-visible expert_demo directory; it must not already exist",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    receipt = project_fixed_demo_bundle(
        asset_root=resolve_fixed_demo_root(args.fixed_demo_root),
        destination=args.output,
        task=args.task,
        profile=get_observation_profile(args.profile),
        annotations=AnnotationCapabilities(args.provide_bbox, args.provide_mask),
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
