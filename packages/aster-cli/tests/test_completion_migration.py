# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Completion upgrades preserve other shell configuration and replace generated setup."""

import importlib

import pytest
from click.testing import CliRunner

completion = importlib.import_module("aster_cli.completion")


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    monkeypatch.setattr(completion.Path, "home", lambda: tmp_path)
    return tmp_path


@pytest.mark.parametrize("shell", ["bash", "zsh"])
@pytest.mark.parametrize("ending", ["\n", "\r\n"])
@pytest.mark.parametrize("previous", ["NOOA", "NEMO"])
def test_install_replaces_previous_setup_and_preserves_unicode_and_newlines(
    isolated_home, monkeypatch, shell, ending, previous
):
    monkeypatch.setenv("SHELL", f"/bin/{shell}")
    profile = isolated_home / f".{shell}rc"
    unrelated = f"# 保留个人设置{ending}export EXAMPLE_VALUE=unchanged{ending}"
    old = (
        f"# {previous.lower()} shell completions{ending}"
        f'eval "$(_{previous}_COMPLETE={shell}_source nooa)"{ending}'
    )
    # Mentions and custom commands are user content, not installer-generated setup.
    note = f"# _NOOA_COMPLETE migration notes{ending}echo _NEMO_COMPLETE{ending}"
    profile.write_bytes((unrelated + old + note).encode())

    first = CliRunner().invoke(completion.install)

    assert first.exit_code == 0, first.output
    expected = (
        unrelated
        + note
        + f"{ending}# aster shell completions{ending}"
        + f'eval "$(_ASTER_COMPLETE={shell}_source aster)"{ending}'
    ).encode()
    assert profile.read_bytes() == expected
    second = CliRunner().invoke(completion.install)
    assert second.exit_code == 0, second.output
    assert profile.read_bytes() == expected


def test_install_removes_previous_setup_even_when_aster_is_already_installed(
    isolated_home, monkeypatch
):
    monkeypatch.setenv("SHELL", "/bin/bash")
    profile = isolated_home / ".bashrc"
    current = 'eval "$(_ASTER_COMPLETE=bash_source aster)"\n'
    profile.write_text(
        'eval "$(_NOOA_COMPLETE=bash_source nooa)"\n' + current,
        encoding="utf-8",
        newline="",
    )

    result = CliRunner().invoke(completion.install)

    assert result.exit_code == 0, result.output
    assert profile.read_bytes() == current.encode()


def test_fish_install_replaces_only_generated_previous_files(isolated_home, monkeypatch):
    monkeypatch.setenv("SHELL", "/bin/fish")
    directory = isolated_home / ".config/fish/completions"
    directory.mkdir(parents=True)
    for name, brand in (("nooa", "NOOA"), ("nemo", "NEMO")):
        script = completion._FISH_SCRIPT % {
            "prog_name": name,
            "complete_func": f"_{name}_completion",
            "complete_var": f"_{brand}_COMPLETE",
        }
        (directory / f"{name}.fish").write_text(script + "\n", encoding="utf-8")
    unrelated = directory / "other.fish"
    unrelated.write_bytes(b"complete -c other -l custom\n")

    result = CliRunner().invoke(completion.install)

    assert result.exit_code == 0, result.output
    assert not (directory / "nooa.fish").exists()
    assert not (directory / "nemo.fish").exists()
    current = (directory / "aster.fish").read_bytes()
    assert b"_ASTER_COMPLETE=fish_complete" in current
    assert unrelated.read_bytes() == b"complete -c other -l custom\n"
    assert CliRunner().invoke(completion.install).exit_code == 0
    assert (directory / "aster.fish").read_bytes() == current


def test_fish_install_preserves_custom_previous_named_files(isolated_home, monkeypatch):
    monkeypatch.setenv("SHELL", "/bin/fish")
    directory = isolated_home / ".config/fish/completions"
    directory.mkdir(parents=True)
    custom = directory / "nemo.fish"
    custom.write_bytes(b"# File manager completion\ncomplete -c nemo -l browser\n")
    aster_custom = directory / "aster.fish"
    own_content = b"# Custom command options\ncomplete -c aster -l user-added\n"
    aster_custom.write_bytes(own_content)

    result = CliRunner().invoke(completion.install)

    assert result.exit_code == 0, result.output
    assert custom.read_bytes() == b"# File manager completion\ncomplete -c nemo -l browser\n"
    assert aster_custom.read_bytes().startswith(own_content)
    assert b"_ASTER_COMPLETE=fish_complete" in aster_custom.read_bytes()
