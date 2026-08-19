#!/usr/bin/env python3
"""Validate every public projection of the registered fixed demonstrations."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent_env.benchmark_profiles import (  # noqa: E402
    AnnotationCapabilities,
    list_observation_profiles,
)
from agent_env.benchmark_tasks import list_benchmark_tasks  # noqa: E402
from agent_env.fixed_demo_bundle import (  # noqa: E402
    project_fixed_demo_bundle,
    resolve_fixed_demo_root,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Project and validate task x P1-P6 x bbox/mask for registered masters"
        )
    )
    parser.add_argument(
        "--fixed-demo-root",
        type=Path,
        help=(
            "Host-side directory containing registered P6 masters; defaults to "
            "UNIVTAC_FIXED_EXPERT_MASTER_ROOT"
        ),
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Omit per-combination receipts from the JSON result",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    asset_root = resolve_fixed_demo_root(args.fixed_demo_root)
    results: list[dict[str, object]] = []
    for task in list_benchmark_tasks():
        for profile in list_observation_profiles():
            for provide_bbox, provide_mask in (
                (False, False),
                (True, False),
                (False, True),
                (True, True),
            ):
                annotations = AnnotationCapabilities(provide_bbox, provide_mask)
                with tempfile.TemporaryDirectory(
                    prefix="univtac-fixed-demo-validation-"
                ) as temporary:
                    receipt = project_fixed_demo_bundle(
                        asset_root=asset_root,
                        destination=Path(temporary) / "expert_demo",
                        task=task.name,
                        profile=profile,
                        annotations=annotations,
                    )
                results.append(
                    {
                        "task": task.name,
                        "profile": profile.name,
                        "bbox": provide_bbox,
                        "mask": provide_mask,
                        "manifest_sha256": receipt["agent_bundle"][
                            "manifest_sha256"
                        ],
                        "content_integrity": receipt["agent_bundle"][
                            "content_integrity"
                        ],
                    }
                )
    summary = {
        "schema_version": "univtac.fixed_demo_projection_matrix_validation.v1",
        "passed": True,
        "combination_count": len(results),
    }
    if not args.summary_only:
        summary["results"] = results
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
