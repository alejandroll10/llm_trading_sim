"""Shared test bootstrap for the suite.

Two things every test file previously had to hand-roll (and several forgot or
did divergently):

1. Import path: production code lives under src/ and uses absolute imports
   (`from market...`, `from agents...`), so src/ must be on sys.path before
   any test module imports.

2. LoggingService stub: most unit tests construct engine objects without the
   full LoggingService initialization, so `services.logging_service` is
   replaced with a stub before production modules import it.

Both live in tests/_logging_stub.py so that files which also run standalone
(`python tests/x.py`, where conftest.py never loads) can install the same
stub. Do NOT reintroduce per-file stub copies: production modules snapshot
the class at first import, so competing stubs made the winner depend on
collection order.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _logging_stub

_logging_stub.install()
