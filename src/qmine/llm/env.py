"""Loading API keys from a ``.env`` file.

Environment variables are the right place for secrets and the wrong place for
convenience: they vanish between shells, they are invisible to a process the
user did not launch themselves, and "I exported it" and "the program can see it"
are different claims that look identical from the outside.

So keys are read from a ``.env`` file as well, searched upward from the working
directory. Values are never logged, never written into artifacts, and never
placed in a run manifest — only the *name* of the variable that was found.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

log = logging.getLogger("qmine.env")

_LINE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)\s*$")


def load_dotenv(start: str | Path | None = None, *, override: bool = False) -> dict[str, str]:
    """Load ``.env`` from ``start`` or the nearest parent that has one.

    Returns the variable *names* that were set, never their values. Existing
    environment variables win unless ``override`` — a key exported deliberately
    in the shell should not be silently replaced by a stale file.
    """
    here = Path(start or Path.cwd()).resolve()
    candidates = [here, *here.parents]
    loaded: dict[str, str] = {}

    for d in candidates:
        env_file = d / ".env"
        if not env_file.exists():
            continue
        try:
            text = env_file.read_text(encoding="utf-8")
        except OSError as exc:
            log.warning("could not read %s: %s", env_file, exc)
            break
        for raw in text.splitlines():
            if not raw.strip() or raw.lstrip().startswith("#"):
                continue
            m = _LINE.match(raw)
            if not m:
                continue
            name, value = m.group(1), m.group(2).strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            if name in os.environ and not override:
                continue
            os.environ[name] = value
            loaded[name] = str(env_file)
        break

    if loaded:
        log.info("loaded %d variable(s) from %s", len(loaded), next(iter(loaded.values())))
    return loaded


def key_status() -> dict[str, bool]:
    """Which known provider variables are set.  Booleans only — never values."""
    from .providers import PROVIDERS

    return {
        var: bool(os.environ.get(var))
        for spec in PROVIDERS
        for var in spec.env_vars
    }
