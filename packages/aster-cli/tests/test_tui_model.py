# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Aster defaults use the backend's workspace and explicit configuration layers."""

import os
from pathlib import Path

import pytest
from aster_cli._common import find_project_root
from aster_cli._tui_model import _config_paths, resolve_tui_model


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    selected = tmp_path / "selected workspace"
    selected.mkdir()
    caller = tmp_path / "caller"
    caller.mkdir()
    monkeypatch.chdir(caller)
    monkeypatch.setenv("ASTER_USER_DIR", str(tmp_path / "user"))
    for name in (
        "ASTER_PROJECT_DIR",
        "ASTER_SETTINGS",
        "ASTER_LLM_CONFIG",
        "XDG_CONFIG_HOME",
    ):
        monkeypatch.delenv(name, raising=False)
    return selected


def write_config(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def test_project_default_overrides_user_and_does_not_need_model_registry(workspace, tmp_path):
    write_config(tmp_path / "user/settings.yaml", "tui:\n  default_model: user-model\n")
    write_config(workspace / ".aster/settings.yaml", "tui:\n  default_model: project-model\n")
    write_config(workspace / ".aster/llm_config.yaml", "models: [invalid YAML")

    assert resolve_tui_model(workspace) == "project-model"


def test_user_default_is_used_without_project_default(workspace, tmp_path):
    write_config(tmp_path / "user/settings.yaml", "tui:\n  default_model: user-model\n")
    write_config(workspace / ".aster/settings.yaml", "tui:\n  theme: dark\n")

    assert resolve_tui_model(workspace) == "user-model"


def test_settings_overrides_are_workspace_relative_and_last_wins(workspace, tmp_path, monkeypatch):
    write_config(tmp_path / "user/settings.yaml", "tui:\n  default_model: user-model\n")
    write_config(workspace / ".aster/settings.yaml", "tui:\n  default_model: project-model\n")
    write_config(workspace / "first.yaml", "tui:\n  default_model: first-model\n")
    write_config(workspace / "second.yaml", "tui:\n  default_model: last-model\n")
    write_config(Path.cwd() / "second.yaml", "tui:\n  default_model: wrong-cwd-model\n")
    monkeypatch.setenv("ASTER_SETTINGS", " first.yaml, , missing.yaml, second.yaml ")

    assert resolve_tui_model(workspace) == "last-model"


def test_repeated_config_paths_keep_the_last_layer_position(workspace, tmp_path, monkeypatch):
    user = write_config(tmp_path / "user/settings.yaml", "tui:\n  default_model: user-model\n")
    project = write_config(
        workspace / ".aster/settings.yaml", "tui:\n  default_model: project-model\n"
    )
    monkeypatch.setenv("ASTER_SETTINGS", f"{user},{project},{user}")

    assert _config_paths(workspace, "settings.yaml", "ASTER_SETTINGS") == [project, user]
    assert resolve_tui_model(workspace) == "user-model"


@pytest.mark.parametrize(
    "replacement", ["tui: null\n", "tui: disabled\n", "tui:\n  default_model: null\n"]
)
def test_settings_deletion_clears_earlier_default(workspace, tmp_path, replacement):
    write_config(tmp_path / "user/settings.yaml", "tui:\n  default_model: old-default\n")
    write_config(workspace / ".aster/settings.yaml", replacement)
    write_config(workspace / ".aster/llm_config.yaml", "models:\n  only-chat: {}\n")

    assert resolve_tui_model(workspace) == "only-chat"


@pytest.mark.parametrize("value", ["''", "'   '", "42", "[]"])
def test_invalid_explicit_default_is_not_replaced_by_an_inferred_alias(workspace, value):
    write_config(workspace / ".aster/settings.yaml", f"tui:\n  default_model: {value}\n")
    write_config(workspace / ".aster/llm_config.yaml", "models:\n  only-chat: {}\n")

    with pytest.raises(ValueError, match="tui.default_model.*nonblank"):
        resolve_tui_model(workspace)


def test_default_accepts_a_model_name_without_a_registered_alias(workspace):
    write_config(workspace / ".aster/settings.yaml", "tui:\n  default_model: openai/my-model\n")

    assert resolve_tui_model(workspace) == "openai/my-model"


def test_selected_workspace_uses_nearest_project_root(workspace):
    write_config(workspace / "pyproject.toml", "[project]\n")
    write_config(workspace / ".aster/settings.yaml", "tui:\n  default_model: selected-project\n")
    write_config(Path.cwd() / "pyproject.toml", "[project]\n")
    write_config(Path.cwd() / ".aster/settings.yaml", "tui:\n  default_model: caller-project\n")
    nested = workspace / "src/nested"
    nested.mkdir(parents=True)

    assert resolve_tui_model(nested) == "selected-project"
    assert find_project_root(nested) == workspace
    assert find_project_root() == Path.cwd()


def test_project_directory_override_is_relative_to_selected_workspace(workspace, monkeypatch):
    write_config(workspace / ".aster/settings.yaml", "tui:\n  default_model: ordinary-project\n")
    write_config(workspace / "custom/settings.yaml", "tui:\n  default_model: override-project\n")
    write_config(Path.cwd() / "custom/settings.yaml", "tui:\n  default_model: wrong-cwd-model\n")
    monkeypatch.setenv("ASTER_PROJECT_DIR", "custom")

    assert resolve_tui_model(workspace) == "override-project"


@pytest.mark.parametrize(
    ("variable", "directory"),
    [("ASTER_USER_DIR", "custom-user"), ("XDG_CONFIG_HOME", "custom-user/aster")],
)
def test_relative_user_directories_follow_backend_cwd(workspace, monkeypatch, variable, directory):
    monkeypatch.delenv("ASTER_USER_DIR")
    monkeypatch.setenv(variable, "custom-user")
    write_config(workspace / directory / "settings.yaml", "tui:\n  default_model: selected-user\n")
    write_config(Path.cwd() / directory / "settings.yaml", "tui:\n  default_model: caller-user\n")

    assert resolve_tui_model(workspace) == "selected-user"


def test_user_directory_override_wins_over_xdg(workspace, tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    write_config(tmp_path / "xdg/aster/settings.yaml", "tui:\n  default_model: xdg-model\n")
    write_config(tmp_path / "user/settings.yaml", "tui:\n  default_model: explicit-user\n")

    assert resolve_tui_model(workspace) == "explicit-user"


def test_user_directory_falls_back_to_home_config(workspace, tmp_path, monkeypatch):
    monkeypatch.delenv("ASTER_USER_DIR")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))
    write_config(
        tmp_path / "home/.config/aster/settings.yaml", "tui:\n  default_model: home-model\n"
    )

    assert resolve_tui_model(workspace) == "home-model"


def test_environment_file_paths_expand_tilde(workspace, tmp_path, monkeypatch):
    original_expanduser = os.path.expanduser

    def expanduser(value):
        if value == "~" or value.startswith(("~/", "~\\")):
            return str(tmp_path / "home") + value[1:]
        return original_expanduser(value)

    monkeypatch.setattr(os.path, "expanduser", expanduser)
    write_config(tmp_path / "home/model-settings.yaml", "tui:\n  default_model: home-override\n")
    monkeypatch.setenv("ASTER_SETTINGS", "~/model-settings.yaml")

    assert resolve_tui_model(workspace) == "home-override"


@pytest.mark.parametrize("directory_var", ["ASTER_USER_DIR", "ASTER_PROJECT_DIR"])
def test_directory_overrides_preserve_literal_tilde(workspace, monkeypatch, directory_var):
    monkeypatch.setenv(directory_var, "~/custom")
    write_config(workspace / "~/custom/settings.yaml", "tui:\n  default_model: literal-directory\n")

    assert resolve_tui_model(workspace) == "literal-directory"


def test_user_singleton_alias_is_inferred(workspace, tmp_path):
    write_config(tmp_path / "user/llm_config.yaml", "models:\n  user-chat: {}\n")

    assert resolve_tui_model(workspace) == "user-chat"


def test_separate_user_and_project_aliases_are_ambiguous(workspace, tmp_path):
    write_config(tmp_path / "user/llm_config.yaml", "models:\n  user-chat: {}\n")
    write_config(workspace / ".aster/llm_config.yaml", "models:\n  project-chat: {}\n")

    with pytest.raises(ValueError, match="Multiple configured chat models") as result:
        resolve_tui_model(workspace)
    assert "user-chat" in str(result.value)
    assert "project-chat" in str(result.value)
    assert "tui.default_model" in str(result.value)
    assert "--model" in str(result.value)


def test_later_alias_replaces_the_whole_entry(workspace, tmp_path):
    write_config(tmp_path / "user/llm_config.yaml", "models:\n  model: {client_type: decision}\n")
    write_config(workspace / ".aster/llm_config.yaml", "models:\n  model: {}\n")

    assert resolve_tui_model(workspace) == "model"


@pytest.mark.parametrize("replacement", ["null", "invalid", "{client_type: decision}"])
def test_removed_or_ineligible_alias_does_not_remain_a_candidate(workspace, tmp_path, replacement):
    write_config(tmp_path / "user/llm_config.yaml", "models:\n  removed: {}\n  remaining: {}\n")
    write_config(workspace / ".aster/llm_config.yaml", f"models:\n  removed: {replacement}\n")

    assert resolve_tui_model(workspace) == "remaining"


@pytest.mark.parametrize("models", ["null", "[]", "invalid"])
def test_nonmapping_models_section_does_not_delete_lower_aliases(workspace, tmp_path, models):
    write_config(tmp_path / "user/llm_config.yaml", "models:\n  user-chat: {}\n")
    write_config(workspace / ".aster/llm_config.yaml", f"models: {models}\n")

    assert resolve_tui_model(workspace) == "user-chat"


def test_only_chat_compatible_aliases_are_candidates(workspace):
    write_config(
        workspace / ".aster/llm_config.yaml",
        "models:\n"
        "  decisions: {client_type: decision}\n"
        "  unknown: {client_type: unsupported}\n"
        "  null-client: {client_type: null}\n"
        "  malformed: {client_type: []}\n"
        "  chat: {client_type: responses}\n",
    )

    assert resolve_tui_model(workspace) == "chat"


def test_model_file_overrides_are_workspace_relative_and_last_wins(
    workspace, tmp_path, monkeypatch
):
    write_config(tmp_path / "user/llm_config.yaml", "models:\n  user-chat: {}\n")
    write_config(workspace / ".aster/llm_config.yaml", "models:\n  project-chat: {}\n")
    write_config(workspace / "first.yaml", "models:\n  user-chat: null\n")
    write_config(workspace / "last.yaml", "models:\n  project-chat: null\n  selected-chat: {}\n")
    write_config(Path.cwd() / "last.yaml", "models:\n  wrong-chat: {}\n")
    monkeypatch.setenv("ASTER_LLM_CONFIG", "first.yaml, last.yaml")

    assert resolve_tui_model(workspace) == "selected-chat"


def test_no_models_reports_how_to_set_a_default(workspace):
    with pytest.raises(ValueError, match="No default or configured chat model") as result:
        resolve_tui_model(workspace)

    assert "ASTER_MODEL" in str(result.value)
    assert "tui.default_model" in str(result.value)


@pytest.mark.parametrize("filename", ["settings.yaml", "llm_config.yaml"])
def test_malformed_yaml_reports_path_without_file_contents(workspace, filename):
    path = write_config(workspace / ".aster" / filename, "sensitive_test_value: [unfinished\n")

    with pytest.raises(ValueError, match="Cannot read model configuration") as result:
        resolve_tui_model(workspace)

    assert str(path) in str(result.value)
    assert "sensitive_test_value" not in str(result.value)
    assert "--model" in str(result.value)
