# aster-memory

Long-term memory subsystem for [Aster](../../README.md)
agents. Installs the `aster.memory` skill, which gives an agent persistent recall
across sessions backed by vector search.

```bash
uv sync --extra memory    # from the repository root
```

> **Windows:** use [WSL 2](https://learn.microsoft.com/windows/wsl/install). Native Windows is not supported; see the [main README](../../README.md#installation).

The default vector backend is numpy-only (no extra dependencies). `sqlite-vec`
and `chromadb` backends are imported lazily if you install them yourself.

See the [main repository](../../README.md) for
documentation, and `examples/arc_agi_3` for a worked usage example.

Apache-2.0 licensed.
