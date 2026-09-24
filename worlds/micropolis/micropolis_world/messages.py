"""Warnings and errors, colored, on stderr.

A long prompting run scrolls hundreds of progress lines, so a failed call has
to stand out and has to be separable from the report — hence stderr rather
than stdout, and color rather than plain. Everything that is not a warning or
an error stays on stdout as ordinary output.

Warnings are yellow, errors red and retries blue, so the three are told apart
at a glance rather than by reading the prefix: a retry means nothing is wrong
yet, a warning means the run went on, an error means something did not happen
at all, and only the last is worth interrupting a long run for.

Color is dropped when stderr is not a terminal, so a redirected log holds no
escape sequences, and when NO_COLOR is set (https://no-color.org).
"""

import os
import sys

YELLOW = "\033[33m"
RED = "\033[1;31m"
BLUE = "\033[34m"
RESET = "\033[0m"


def color_enabled(stream=None) -> bool:
    """Whether to emit ANSI codes on `stream` (default stderr)."""
    stream = stream or sys.stderr
    if os.environ.get("NO_COLOR"):
        return False
    return bool(getattr(stream, "isatty", lambda: False)())


def _emit(text: str, prefix: str, color: str) -> None:
    body = f"{prefix}{text}"
    if color_enabled():
        body = f"{color}{body}{RESET}"
    print(body, file=sys.stderr, flush=True)


def warn(text: str) -> None:
    """One warning line on stderr, yellow: something is off, the run goes on."""
    _emit(text, "[warning] ", YELLOW)


def error(text: str) -> None:
    """One error line on stderr, bold red: something did not happen at all."""
    _emit(text, "[error] ", RED)


def retry(text: str) -> None:
    """One retry line on stderr, blue: a call failed transiently and is retried."""
    _emit(text, "[retry] ", BLUE)


def plain(text: str, color: str = RED) -> None:
    """A continuation line on stderr, unprefixed, in `color`.

    For the detail lines under a warning or error — a per-failure entry in a
    summary — which read as part of it rather than as messages of their own.
    Defaults to the error color, since that is where summaries are used; pass
    YELLOW under a warning so the block stays one color.
    """
    _emit(text, "", color)
