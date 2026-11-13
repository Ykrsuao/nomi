# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ATIF step timestamps must be real UTC, not local time wearing a Z suffix.

``_iso8601_utc`` used to format whatever tzinfo it was handed and append ``Z``
unconditionally.  Event timestamps arrive naive (local), while the no-argument
default is already UTC-aware, so a single trajectory ended up mixing both and
its steps were not in chronological order.
"""

import os
import time
from datetime import UTC, datetime, timedelta, timezone

import pytest

from aster.atif.exporter import _iso8601_utc


def _hour(stamp: str) -> int:
    return int(stamp[11:13])


class TestIso8601Utc:
    def test_aware_utc_is_unchanged(self):
        dt = datetime(2026, 7, 30, 10, 6, 42, 680_000, tzinfo=UTC)
        assert _iso8601_utc(dt) == "2026-07-30T10:06:42.680Z"

    def test_aware_non_utc_is_converted(self):
        """A +02:00 wall clock of 12:06 is 10:06 UTC."""
        dt = datetime(2026, 7, 30, 12, 6, 42, 680_000, tzinfo=timezone(timedelta(hours=2)))
        assert _iso8601_utc(dt) == "2026-07-30T10:06:42.680Z"

    def test_negative_offset_is_converted(self):
        dt = datetime(2026, 7, 30, 6, 6, 42, 680_000, tzinfo=timezone(timedelta(hours=-4)))
        assert _iso8601_utc(dt) == "2026-07-30T10:06:42.680Z"

    def test_naive_is_interpreted_as_local(self):
        """Naive event timestamps are local time and must be shifted to UTC."""
        naive = datetime.now()
        assert _iso8601_utc(naive) == _iso8601_utc(naive.astimezone(UTC))

    def test_default_matches_explicit_now(self):
        assert abs(
            datetime.strptime(_iso8601_utc(), "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=UTC)
            - datetime.now(UTC)
        ) < timedelta(seconds=5)

    @pytest.mark.skipif(not hasattr(time, "tzset"), reason="tzset() is POSIX-only")
    def test_naive_and_default_agree_under_a_non_utc_tz(self):
        """The regression: two code paths must not disagree by the UTC offset.

        Under a non-UTC local zone, an event-sourced (naive) timestamp and the
        no-argument default previously landed hours apart in the same file.
        """
        prev = os.environ.get("TZ")
        try:
            os.environ["TZ"] = "Europe/Berlin"  # UTC+2 in July
            time.tzset()
            from_event = _iso8601_utc(datetime.now())  # naive → local
            from_default = _iso8601_utc()  # aware → UTC
            assert _hour(from_event) == _hour(from_default)
        finally:
            if prev is None:
                os.environ.pop("TZ", None)
            else:
                os.environ["TZ"] = prev
            time.tzset()
