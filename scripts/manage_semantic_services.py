#!/usr/bin/env python3
"""Start, stop, and inspect the isolated SAM3/UniDepth V2 services."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPO_ROOT.parent
DEFAULT_STATE_DIR = REPO_ROOT / ".semantic_runtime"
DEFAULT_MODEL_ROOT = REPO_ROOT / ".semantic_models"


@dataclass(frozen=True)
class ServiceConfig:
    name: str
    module: str
    python: Path
    port: int
    arguments: tuple[str, ...]

    @property
    def health_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/health"


def _lock() -> dict[str, Any]:
    return json.loads(
        (REPO_ROOT / "semantic_tools" / "versions.json").read_text(encoding="utf-8")
    )


def service_configs(args: argparse.Namespace) -> dict[str, ServiceConfig]:
    lock = _lock()
    model_root = args.model_root.resolve()
    sam = lock["sam3"]
    depth = lock["unidepth_v2"]
    return {
        "sam3": ServiceConfig(
            name="sam3",
            module="semantic_tools.sam3_server",
            python=Path(args.sam3_python).resolve(),
            port=args.sam3_port,
            arguments=(
                "--checkpoint",
                str(model_root / "sam3" / "sam3.pt"),
                "--revision",
                sam["model_revision"],
                "--device",
                args.device,
                "--port",
                str(args.sam3_port),
            ),
        ),
        "unidepth_v2": ServiceConfig(
            name="unidepth_v2",
            module="semantic_tools.unidepth_v2_server",
            python=Path(args.unidepth_v2_python).resolve(),
            port=args.unidepth_v2_port,
            arguments=(
                "--model-dir",
                str(model_root / "unidepth-v2-vitl14"),
                "--revision",
                depth["model_revision"],
                "--device",
                args.device,
                "--port",
                str(args.unidepth_v2_port),
            ),
        ),
    }


def _paths(state_dir: Path, name: str) -> tuple[Path, Path]:
    return state_dir / f"{name}.json", state_dir / f"{name}.log"


def _read_state(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _pid_matches(state: dict[str, Any], config: ServiceConfig) -> bool:
    pid = state.get("pid")
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 1:
        return False
    proc = Path("/proc") / str(pid)
    try:
        cmdline = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode()
        actual_start = int((proc / "stat").read_text().split()[21])
    except (OSError, ValueError, IndexError, UnicodeDecodeError):
        return False
    return (
        state.get("start_ticks") == actual_start
        and config.module in cmdline
        and f"--port {config.port}" in cmdline
    )


def _health(config: ServiceConfig, timeout: float = 2.0) -> dict[str, Any] | None:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(config.health_url, timeout=timeout) as response:
            value = json.loads(response.read(1024 * 1024))
    except (OSError, urllib.error.URLError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) and value.get("success") is True else None


def _environment(args: argparse.Namespace) -> dict[str, str]:
    env = dict(os.environ)
    for key in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ):
        env.pop(key, None)
    env["PYTHONPATH"] = str(REPO_ROOT)
    env["HF_HOME"] = str(args.hf_home.resolve())
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def start(config: ServiceConfig, args: argparse.Namespace) -> dict[str, Any]:
    args.state_dir.mkdir(parents=True, exist_ok=True)
    state_path, log_path = _paths(args.state_dir, config.name)
    existing = _read_state(state_path)
    if _pid_matches(existing, config):
        health = _health(config)
        return {"name": config.name, "running": True, "already_running": True, "health": health}
    if not config.python.is_file() or not os.access(config.python, os.X_OK):
        raise RuntimeError(f"service Python is missing: {config.python}")
    command = [str(config.python), "-m", config.module, *config.arguments]
    with log_path.open("ab", buffering=0) as log:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=_environment(args),
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    start_ticks = int((Path("/proc") / str(process.pid) / "stat").read_text().split()[21])
    state = {
        "schema_version": "univtac.semantic_service_process.v1",
        "name": config.name,
        "pid": process.pid,
        "start_ticks": start_ticks,
        "port": config.port,
        "module": config.module,
        "log": log_path.name,
    }
    state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    deadline = time.monotonic() + args.start_timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
            raise RuntimeError(f"{config.name} exited during startup:\n{tail}")
        health = _health(config)
        if health is not None:
            return {"name": config.name, "running": True, "already_running": False, "health": health}
        time.sleep(0.25)
    stop(config, args)
    raise RuntimeError(f"{config.name} did not become healthy within {args.start_timeout}s")


def stop(config: ServiceConfig, args: argparse.Namespace) -> dict[str, Any]:
    state_path, _ = _paths(args.state_dir, config.name)
    state = _read_state(state_path)
    if not _pid_matches(state, config):
        state_path.unlink(missing_ok=True)
        return {"name": config.name, "stopped": True, "was_running": False}
    pid = int(state["pid"])
    os.killpg(pid, signal.SIGTERM)
    deadline = time.monotonic() + args.stop_timeout
    while time.monotonic() < deadline and _pid_matches(state, config):
        time.sleep(0.1)
    if _pid_matches(state, config):
        os.killpg(pid, signal.SIGKILL)
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and _pid_matches(state, config):
            time.sleep(0.1)
    if _pid_matches(state, config):
        raise RuntimeError(f"could not stop {config.name} pid {pid}")
    state_path.unlink(missing_ok=True)
    return {"name": config.name, "stopped": True, "was_running": True}


def status(config: ServiceConfig, args: argparse.Namespace) -> dict[str, Any]:
    state_path, log_path = _paths(args.state_dir, config.name)
    state = _read_state(state_path)
    running = _pid_matches(state, config)
    return {
        "name": config.name,
        "running": running,
        "pid": state.get("pid") if running else None,
        "port": config.port,
        "health": _health(config) if running else None,
        "log": str(log_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("start", "stop", "restart", "status", "health"))
    parser.add_argument("service", choices=("sam3", "unidepth_v2", "all"))
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument(
        "--sam3-python",
        default=str(PROJECT_ROOT / "miniconda3" / "envs" / "univtac-sam3" / "bin" / "python"),
    )
    parser.add_argument(
        "--unidepth-v2-python",
        default=str(PROJECT_ROOT / "miniconda3" / "envs" / "univtac-unidepth-v2" / "bin" / "python"),
    )
    parser.add_argument("--hf-home", type=Path, default=Path("/inspire/qb-ilm/project/semantic-visual-tokenizer/public/dzj/univtac_semantic_tools/hf"))
    parser.add_argument("--sam3-port", type=int, default=8783)
    parser.add_argument("--unidepth-v2-port", type=int, default=8784)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--start-timeout", type=float, default=30.0)
    parser.add_argument("--stop-timeout", type=float, default=20.0)
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.state_dir = args.state_dir.resolve()
    configs = service_configs(args)
    names = list(configs) if args.service == "all" else [args.service]
    results: list[dict[str, Any]] = []
    try:
        for name in names:
            config = configs[name]
            if args.action == "start":
                results.append(start(config, args))
            elif args.action == "stop":
                results.append(stop(config, args))
            elif args.action == "restart":
                stop(config, args)
                # Give the terminated HTTP server a short interval to release
                # its listening socket before binding the same port again.
                time.sleep(0.5)
                results.append(start(config, args))
            else:
                item = status(config, args)
                results.append(item)
                if args.action == "health" and item["health"] is None:
                    raise RuntimeError(f"{name} health check failed")
    except Exception as exc:
        payload = {"success": False, "error_type": type(exc).__name__, "message": str(exc), "services": results}
        print(json.dumps(payload, ensure_ascii=False, indent=2) if args.json else f"ERROR: {exc}", file=sys.stderr)
        return 1
    payload = {"success": True, "services": results}
    print(json.dumps(payload, ensure_ascii=False, indent=2) if args.json else "\n".join(str(item) for item in results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
