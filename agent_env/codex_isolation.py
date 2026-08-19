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

_PRIVATE_BENCHMARK_ENVIRONMENT = {
    "PYTHONPATH",
    "UNIVTAC_EVALUATOR_SEED",
    "UNIVTAC_FIXED_EXPERT_MASTER_ROOT",
}


class IsolatedCodexEnvironment(AbstractContextManager["IsolatedCodexEnvironment"]):
    """Create a fresh workspace with selectable Codex configuration inheritance."""

    def __init__(
        self,
        auth_home: Path | None = None,
        *,
        inherit_agent_configuration: bool = False,
    ) -> None:
        source_home = auth_home
        if source_home is None:
            configured = os.environ.get("CODEX_HOME")
            source_home = Path(configured) if configured else Path.home() / ".codex"
        self.source_home = source_home.expanduser().resolve()
        self.inherit_agent_configuration = inherit_agent_configuration
        self.root = Path(tempfile.mkdtemp(prefix="univtac-codex-isolated-"))
        self.home = self.root / "home"
        self.codex_home = (
            self.source_home
            if inherit_agent_configuration
            else self.home / ".codex"
        )
        self.workspace = self.root / "workspace"
        self.tmp = self.root / "tmp"
        local_directories = [self.home, self.workspace, self.tmp]
        if not inherit_agent_configuration:
            local_directories.append(self.codex_home)
        for path in local_directories:
            path.mkdir(mode=0o700, parents=True, exist_ok=True)

        source_auth = self.source_home / "auth.json"
        if not source_auth.is_file():
            self.close()
            raise FileNotFoundError(
                f"Codex authentication file not found at {source_auth}. Run `codex login`."
            )
        if not inherit_agent_configuration:
            target_auth = self.codex_home / "auth.json"
            shutil.copyfile(source_auth, target_auth)
            target_auth.chmod(0o600)

    def child_environment(self) -> dict[str, str]:
        if self.inherit_agent_configuration:
            environment = dict(os.environ)
            for key in _PRIVATE_BENCHMARK_ENVIRONMENT:
                environment.pop(key, None)
            environment.update(
                {
                    "CODEX_HOME": str(self.codex_home),
                    "TMPDIR": str(self.tmp),
                }
            )
        else:
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
        inherited = self.inherit_agent_configuration
        return {
            "fresh_codex_home": not inherited,
            "fresh_empty_workspace": True,
            "auth_source_path_recorded": False,
            "copied_authentication_only": [] if inherited else ["auth.json"],
            "inherited_config": inherited,
            "inherited_sessions": inherited,
            "inherited_memories": inherited,
            "inherited_mcp_servers": inherited,
            "inherited_plugins": inherited,
            "inherited_skills": inherited,
            "private_benchmark_environment_removed": sorted(
                _PRIVATE_BENCHMARK_ENVIRONMENT
            ),
            "temporary_paths_removed_after_run": True,
            "child_environment_keys": sorted(self.child_environment()),
        }

    def close(self) -> None:
        if self.root.exists():
            shutil.rmtree(self.root)

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def embodied_codex_command(
    codex_bin: str = "codex",
    *,
    enable_general_capabilities: bool = False,
) -> list[str]:
    """Return an app-server command for legacy restricted or general runs."""

    command = [codex_bin, "app-server", "--stdio"]
    if enable_general_capabilities:
        return command

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
    for feature in disabled_features:
        command.extend(("--disable", feature))
    return command
