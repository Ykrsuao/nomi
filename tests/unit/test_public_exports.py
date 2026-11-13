# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Names hosts import from public modules instead of private ones."""

import aster.runtime.channels as channels
import aster.tools
from aster.tools._bash_session import BashSession as PrivateBashSession
from aster.tools._results import StreamDone as PrivateStreamDone
from aster.tools._results import StreamEvent as PrivateStreamEvent


def test_tools_exports_bash_session_and_stream_types():
    from aster.tools import BashSession, StreamDone, StreamEvent

    assert BashSession is PrivateBashSession
    assert StreamEvent is PrivateStreamEvent
    assert StreamDone is PrivateStreamDone
    for name in ("BashSession", "StreamEvent", "StreamDone"):
        assert name in aster.tools.__all__


def test_channels_exports_channel_reader():
    from aster.runtime.channels import ChannelReader

    ch = channels.Channel("q", "queue")
    assert isinstance(ch.reader, ChannelReader)
    assert ChannelReader.__name__ == "ChannelReader"
