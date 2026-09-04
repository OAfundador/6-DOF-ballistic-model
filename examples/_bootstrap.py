"""Make ``sixdof`` importable when the repository has not been installed.

The examples are meant to run straight out of a fresh clone -- ``python
examples/01_single_shot.py`` -- so each of them imports this module first.  If
the package *is* installed (``pip install -e .``), the inserted path is simply
redundant.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"


def ensure_package_on_path() -> Path:
    """Prepend ``src/`` to ``sys.path`` and return the repository root."""
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    return REPO_ROOT


def configure_stdout() -> None:
    """Switch stdout/stderr to UTF-8 where the console defaults to something else.

    Needed on Windows, where the reports below would otherwise fail on the
    accented characters and the box-drawing rules.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):  # pragma: no cover - platform dependent
            pass


__all__ = ["ensure_package_on_path", "configure_stdout", "REPO_ROOT", "SRC"]
