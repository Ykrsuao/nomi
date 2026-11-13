# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ACP startup must not wait for an unrelated remote pricing catalogue."""

import json
import os
import subprocess
import sys

import pytest

_STARTUP_PROBE = """
import json
import httpx

downloads = []

def unavailable_catalogue(url, *args, **kwargs):
    downloads.append(str(url))
    raise httpx.ConnectError('Synthetic unavailable pricing catalogue')

httpx.get = unavailable_catalogue

from click.testing import CliRunner
from aster_acp import server
from aster_acp.cli import command

clients = []

async def prepare_model(factory, **options):
    client = factory()
    clients.append(client)
    await client.aclose()

server.serve = prepare_model
result = CliRunner().invoke(command, ['--model', 'startup-fixture'])
assert result.exit_code == 0, result.output
print(json.dumps({
    'download_attempts': len(downloads),
    'catalogue_download_attempts': sum(
        'model_prices_and_context_window.json' in url for url in downloads
    ),
    'prepared_models': len(clients),
}))
"""


@pytest.mark.parametrize("remote_catalogue", [None, "True", "False"])
def test_startup_uses_bundled_catalogue_unless_remote_is_requested(tmp_path, remote_catalogue):
    # A fresh interpreter reaches the real model factory before LiteLLM is loaded.
    # The unavailable remote table reproduces a startup network dependency without
    # contacting a provider or relying on wall-clock timing.
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config = config_dir / "llm_config.yaml"
    config.write_text(
        "models:\n"
        "  startup-fixture:\n"
        "    model_name: openai/startup-fixture\n"
        "    api_base: http://127.0.0.1:9/v1\n"
        "    api_key: synthetic-startup-key\n"
        "    media_inputs: []\n",
        encoding="utf-8",
    )
    environment = dict(os.environ)
    environment.update(
        ASTER_USER_DIR=str(config_dir),
        ASTER_PROJECT_DIR=str(tmp_path),
        ASTER_LLM_CONFIG=str(config),
        PYTHONIOENCODING="utf-8",
    )
    environment.pop("ASTER_SECRETS", None)
    if remote_catalogue is None:
        environment.pop("LITELLM_LOCAL_MODEL_COST_MAP", None)
    else:
        environment["LITELLM_LOCAL_MODEL_COST_MAP"] = remote_catalogue

    result = subprocess.run(
        [sys.executable, "-c", _STARTUP_PROBE],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=45,
        check=True,
    )
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert report["prepared_models"] == 1
    if remote_catalogue == "False":
        # LiteLLM may retry a failed catalogue request in a background thread.
        # Explicit opt-in must fetch the table; its retry count is SDK-owned.
        assert report["catalogue_download_attempts"] >= 1
    else:
        assert report["download_attempts"] == 0
