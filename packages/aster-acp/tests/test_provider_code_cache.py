# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Windows ACP startup reads provider code in bulk without changing SDK behavior."""

import json
import os
import subprocess
import sys

import pytest

_REAL_STARTUP = """
import asyncio
import importlib.machinery
import importlib.resources
import json
import sys

import httpx
from click.testing import CliRunner
from aster_acp import server
from aster_acp.cli import command

assert 'litellm' not in sys.modules
assert 'openai' not in sys.modules
original_get_data = importlib.machinery.SourceFileLoader.get_data

def reject_individual_provider_code(loader, path):
    normalized = path.replace('\\\\', '/').lower()
    if normalized.endswith(('.py', '.pyc')) and any(
        '/site-packages/' + package + '/' in normalized
        for package in ('litellm', 'openai')
    ):
        raise RuntimeError('Individual provider code read: ' + path)
    return original_get_data(loader, path)

importlib.machinery.SourceFileLoader.get_data = reject_individual_provider_code
requests = []

async def model_response(self, request, **options):
    assert request.url.host == '127.0.0.1'
    assert request.url.path == '/v1/chat/completions'
    body = json.loads(request.content)
    assert body['model'] == 'startup-fixture'
    requests.append(body)
    return httpx.Response(200, request=request, json={
        'id': 'fixture-response', 'object': 'chat.completion', 'created': 0,
        'model': 'startup-fixture',
        'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'OK'},
                     'finish_reason': 'stop'}],
        'usage': {'prompt_tokens': 3, 'completion_tokens': 1, 'total_tokens': 4},
    })

httpx.AsyncClient.send = model_response
prepared = []

async def prepare_model(factory, **options):
    client = factory()
    try:
        response = await client.acall([{'role': 'user', 'content': 'Only reply OK.'}])
        assert response.content == 'OK', response
        prepared.append(client)
        import litellm
        import openai
        assert type(litellm.__loader__).__name__ == '_CodeLoader'
        assert type(openai.__loader__).__name__ == '_CodeLoader'
        assert importlib.resources.files('litellm').joinpath(
            'model_prices_and_context_window_backup.json'
        ).is_file()
        import litellm.rust_bridge._native as native
        assert isinstance(native.__loader__, importlib.machinery.ExtensionFileLoader)
    finally:
        await client.aclose()

server.serve = prepare_model
result = CliRunner().invoke(command, ['--model', 'startup-fixture'])
assert result.exit_code == 0, (result.output, repr(result.exception))
print(json.dumps({'prepared_models': len(prepared), 'model_requests': len(requests)}))
"""


@pytest.mark.skipif(sys.platform != "win32", reason="Automatic bulk code loading is Windows-only")
def test_real_startup_and_response_do_not_read_individual_sdk_modules(tmp_path):
    config = tmp_path / "llm_config.yaml"
    config.write_text(
        "models:\n"
        "  startup-fixture:\n"
        "    model_name: openai/startup-fixture\n"
        "    api_base: http://127.0.0.1:9/v1\n"
        "    api_key_env: ASTER_STARTUP_TEST_KEY\n"
        "    retry_config: false\n"
        "    media_inputs: []\n",
        encoding="utf-8",
    )
    environment = dict(os.environ)
    environment.update(
        ASTER_USER_DIR=str(tmp_path),
        ASTER_PROJECT_DIR=str(tmp_path),
        ASTER_LLM_CONFIG=str(config),
        XDG_CACHE_HOME=str(tmp_path / "cache"),
        LITELLM_LOCAL_MODEL_COST_MAP="True",
        ASTER_STARTUP_TEST_KEY="synthetic-startup-key",
        PYTHONIOENCODING="utf-8",
    )
    environment.pop("ASTER_SECRETS", None)
    environment.pop("ASTER_PROVIDER_CODE_CACHE", None)
    result = subprocess.run(
        [sys.executable, "-c", _REAL_STARTUP],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip().splitlines()[-1]) == {
        "prepared_models": 1,
        "model_requests": 1,
    }
