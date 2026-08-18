#!/usr/bin/env python3
"""Launch the read-only AgentEnv/Codex interaction viewer."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent_env.run_viewer import create_server, describe_server_urls  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve AgentEnv/Codex interaction logs as a read-only web UI"
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=REPO_ROOT / "agent_runs",
        help="Directory containing AgentEnv runs (default: repo/agent_runs)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    server = create_server(args.host, args.port, args.runs_root)
    actual_port = int(server.server_address[1])
    print("UniVTAC Agent Run Viewer is ready.", flush=True)
    for line in describe_server_urls(args.host, actual_port):
        print(line, flush=True)
    print(f"Runs root: {args.runs_root.expanduser().resolve()}", flush=True)
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        print("\nStopping viewer.", flush=True)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
