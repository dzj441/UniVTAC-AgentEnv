from __future__ import annotations

import os
from pathlib import Path

from agent_env.codex_isolation import IsolatedCodexEnvironment, embodied_codex_command


def test_isolated_home_copies_auth_only_and_is_removed(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "auth.json").write_text('{"token":"test"}', encoding="utf-8")
    (source / "config.toml").write_text("[mcp_servers.bad]", encoding="utf-8")
    (source / "memories_1.sqlite").write_bytes(b"memory")
    (source / "skills").mkdir()

    isolated = IsolatedCodexEnvironment(source)
    root = isolated.root
    try:
        assert {path.name for path in isolated.codex_home.iterdir()} == {"auth.json"}
        assert isolated.workspace.is_dir()
        assert not any(isolated.workspace.iterdir())
        environment = isolated.child_environment()
        assert environment["CODEX_HOME"] == str(isolated.codex_home)
        assert environment["HOME"] == str(isolated.home)
        assert "UNIVTAC_EVALUATOR_SEED" not in environment
        assert "PYTHONPATH" not in environment
        assert isolated.manifest()["inherited_mcp_servers"] is False
    finally:
        isolated.close()
    assert not root.exists()


def test_embodied_codex_command_disables_non_embodied_builtins() -> None:
    command = embodied_codex_command("/bin/codex")
    assert command[:3] == ["/bin/codex", "app-server", "--stdio"]
    disabled = {
        command[index + 1]
        for index, item in enumerate(command[:-1])
        if item == "--disable"
    }
    assert {
        "shell_tool",
        "unified_exec",
        "multi_agent",
        "apps",
        "browser_use",
        "computer_use",
        "view_image",
        "plugins",
        "skill_search",
    } <= disabled
