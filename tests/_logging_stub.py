"""Single shared LoggingService stub + src/ path bootstrap.

This is the ONLY place the `services.logging_service` stub may be defined.
Production modules snapshot the class at first import (`from
services.logging_service import LoggingService`), so competing per-file stub
copies made the winner depend on import order. One shared class installed
idempotently removes the race entirely.

Under pytest, tests/conftest.py calls install() before any test module
imports. Files that also support standalone execution (`python tests/x.py`)
must call install() themselves at the very top, BEFORE importing any
production module — after a production import has resolved the real
services.logging_service, installing the stub has no effect.
"""
import logging
import sys
import types
from pathlib import Path

_SRC = str(Path(__file__).resolve().parents[1] / "src")


class _NoOpMeta(type):
    def __getattr__(cls, name):  # any missing attribute becomes a no-op callable
        return lambda *args, **kwargs: None


class StubLoggingService(metaclass=_NoOpMeta):
    @staticmethod
    def get_logger(name):
        return logging.getLogger(name)


def install():
    """Idempotently put src/ on sys.path and stub services.logging_service."""
    if _SRC not in sys.path:
        sys.path.insert(0, _SRC)
    stub_module = types.ModuleType("services.logging_service")
    stub_module.LoggingService = StubLoggingService
    sys.modules.setdefault("services.logging_service", stub_module)
