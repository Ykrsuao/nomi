# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bearer-token auth for clients posting to a remote aster viewer.

The viewer refuses writes from non-loopback clients unless
``ASTER_VIEWER_AUTH_TOKEN`` is configured on the server
(:func:`aster.viewer.main._require_viewer_authorization`), and it authenticates
with ``Authorization: Bearer <token>``.

Exporters read the same env var on the client side, so a remote run streams
live by setting one variable at both ends::

    # viewer host
    ASTER_VIEWER_AUTH_TOKEN=<token> aster start-dev

    # machine running the agent
    ASTER_VIEWER_AUTH_TOKEN=<token> OTLP_ENDPOINT=http://<host>:5001/v1/traces ...

Unset (the default, loopback-only case) adds no header, so local development is
unchanged.
"""

from __future__ import annotations

import os

_ENV_VAR = "ASTER_VIEWER_AUTH_TOKEN"


def viewer_auth_token() -> str | None:
    """Return the configured viewer token, or ``None`` when unset/blank."""
    token = os.environ.get(_ENV_VAR, "").strip()
    return token or None


def apply_viewer_auth(headers: dict[str, str]) -> dict[str, str]:
    """Add ``Authorization: Bearer`` to *headers* when a token is configured.

    Mutates and returns *headers* so it can be used inline at call sites.
    """
    token = viewer_auth_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers
