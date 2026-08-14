"""Ephemeral Codex home/workspace construction for benchmark rollouts."""

from __future__ import annotations

import os
import shutil
import tempfile
from contextlib import AbstractContextManager
from pathlib import Path


_PASSTHROUGH_ENVIRONMENT = (
    "PATH",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TZ",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "REQUESTS_CA_BUNDLE",
    "CURL_CA_BUNDLE",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "no_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
)


class IsolatedCodexEnvironment(AbstractContextManager["IsolatedCodexEnvironment"]):
    """A fresh home containing authentication but no prior agent state."""

    def __init__(self, auth_home: Path | None = None) -> None:
        source_home = auth_home
        if source_home is None:
            configured = os.environ.get("CODEX_HOME")
            source_home = Path(configured) if configured else Path.home() / ".codex"
        self.source_home = source_home.expanduser().resolve()
        self.root = Path(tempfile.mkdtemp(prefix="univtac-codex-isolated-"))
        self.home = self.root / "home"
        self.codex_home = self.home / ".codex"
        self.workspace = self.root / "workspace"
        self.tmp = self.root / "tmp"
        for path in (self.home, self.codex_home, self.workspace, self.tmp):
            path.mkdir(mode=0o700, parents=True, exist_ok=True)

        source_auth = self.source_home / "auth.json"
        if not source_auth.is_file():
            self.close()
            raise FileNotFoundError(
                f"Codex authentication file not found at {source_auth}. Run `codex login`."
            )
        target_auth = self.codex_home / "auth.json"
        shutil.copyfile(source_auth, target_auth)
        target_auth.chmod(0o600)

    def child_environment(self) -> dict[str, str]:
        environment = {
            key: os.environ[key]
            for key in _PASSTHROUGH_ENVIRONMENT
            if key in os.environ
        }
        environment.update(
            {
                "HOME": str(self.home),
                "CODEX_HOME": str(self.codex_home),
                "TMPDIR": str(self.tmp),
            }
        )
        return environment

    def manifest(self) -> dict[str, object]:
        return {
            "fresh_codex_home": True,
            "fresh_empty_workspace": True,
            "auth_source_path_recorded": False,
            "copied_authentication_only": ["auth.json"],
            "inherited_config": False,
            "inherited_sessions": False,
            "inherited_memories": False,
            "inherited_mcp_servers": False,
            "inherited_plugins": False,
            "inherited_skills": False,
            "temporary_paths_removed_after_run": True,
            "child_environment_keys": sorted(self.child_environment()),
        }

    def close(self) -> None:
        if self.root.exists():
            shutil.rmtree(self.root)

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def embodied_codex_command(codex_bin: str = "codex") -> list[str]:
    """Return an app-server invocation with unrelated built-ins disabled."""

    disabled_features = (
        "shell_tool",
        "unified_exec",
        "multi_agent",
        "apps",
        "browser_use",
        "computer_use",
        "image_generation",
        "view_image",
        "plugins",
        "skill_search",
        "goals",
        "hooks",
        "in_app_browser",
        "browser_use_external",
        "browser_use_full_cdp_access",
        "tool_suggest",
        "workspace_dependencies",
        "shell_snapshot",
    )
    command = [codex_bin, "app-server", "--stdio"]
    for feature in disabled_features:
        command.extend(("--disable", feature))
    return command
