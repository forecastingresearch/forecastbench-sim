"""Tests for the warning/error channel.

The point of these is the routing and the fallback: a warning must never land
on stdout, where it would be mixed into a report, and the ANSI codes must not
reach a redirected log.
"""

import io

from micropolis_world import messages as msg


class FakeTTY(io.StringIO):
    def isatty(self) -> bool:
        return True


def test_warnings_and_errors_go_to_stderr_not_stdout(capsys):
    msg.warn("something is off")
    msg.error("something failed")
    msg.retry("trying again")
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "[warning] something is off" in captured.err
    assert "[error] something failed" in captured.err
    assert "[retry] trying again" in captured.err


def test_no_ansi_codes_when_stderr_is_not_a_terminal(capsys):
    """capsys replaces stderr with a non-tty, which is the redirected case."""
    msg.error("plain please")
    assert "\033[" not in capsys.readouterr().err


def test_colors_when_stderr_is_a_terminal(monkeypatch):
    tty = FakeTTY()
    monkeypatch.setattr("sys.stderr", tty)
    monkeypatch.delenv("NO_COLOR", raising=False)
    msg.warn("warned")
    msg.error("errored")
    msg.retry("retried")
    out = tty.getvalue()
    assert f"{msg.YELLOW}[warning] warned{msg.RESET}" in out
    assert f"{msg.RED}[error] errored{msg.RESET}" in out
    assert f"{msg.BLUE}[retry] retried{msg.RESET}" in out


def test_warnings_errors_and_retries_are_different_colors():
    """The whole point of the split: told apart without reading the prefix."""
    assert len({msg.YELLOW, msg.RED, msg.BLUE}) == 3


def test_no_color_env_var_is_respected(monkeypatch):
    tty = FakeTTY()
    monkeypatch.setattr("sys.stderr", tty)
    monkeypatch.setenv("NO_COLOR", "1")
    msg.error("no escapes")
    assert "\033[" not in tty.getvalue()


def test_plain_carries_no_prefix_for_continuation_lines(capsys):
    """Detail lines under a summary read as part of it, not as new errors."""
    msg.plain("  model <- prompt.txt")
    err = capsys.readouterr().err
    assert err.strip() == "model <- prompt.txt"
    assert "[error]" not in err and "[warning]" not in err


def test_plain_takes_the_color_of_the_block_it_belongs_to(monkeypatch):
    tty = FakeTTY()
    monkeypatch.setattr("sys.stderr", tty)
    monkeypatch.delenv("NO_COLOR", raising=False)
    msg.plain("  under an error")
    msg.plain("  under a warning", msg.YELLOW)
    out = tty.getvalue()
    assert f"{msg.RED}  under an error{msg.RESET}" in out
    assert f"{msg.YELLOW}  under a warning{msg.RESET}" in out
