import subprocess
from pathlib import Path

import pytest

from crib_monitor import probe

URL = "rtsp://camuser:hunter2@cam:554/stream1"


@pytest.fixture
def full(tmp_path, monkeypatch):
    """Run `crib-monitor-probe --full` with subprocess.run replaced by `fake`."""
    monkeypatch.setenv("TAPO_RTSP_URL", URL)
    monkeypatch.setattr(probe, "build_classifiers", lambda *a: pytest.fail("--full must not classify"))
    calls = []

    def run(fake):
        def fake_run(cmd, **kwargs):
            calls.append((cmd, kwargs))
            return fake(cmd)

        monkeypatch.setattr(probe.subprocess, "run", fake_run)
        argv = ["--full", "--out", str(tmp_path / "full.jpg"), "--env-file", str(tmp_path / "none"),
                "--config", str(tmp_path / "missing.toml")]
        with pytest.raises(SystemExit) as exit_:
            probe.run(argv)
        return exit_.value.code, calls

    return run


def test_full_saves_one_uncropped_frame(full, tmp_path, capsys):
    def ok(cmd):
        Path(cmd[-1]).write_bytes(b"\xff\xd8jpeg")
        return subprocess.CompletedProcess(cmd, 0, b"", b"")

    code, calls = full(ok)
    assert code == 0
    (cmd, kwargs), = calls
    assert cmd[0] == "ffmpeg" and URL in cmd
    assert "-vf" not in cmd and not any("crop=" in arg for arg in cmd)
    assert cmd[cmd.index("-frames:v") + 1] == "1"
    assert cmd[-1] == str(tmp_path / "full.jpg")
    assert kwargs["stderr"] == subprocess.PIPE and kwargs["timeout"]
    out = capsys.readouterr()
    assert "full.jpg" in out.out and "hunter2" not in out.out + out.err


def test_full_redacts_ffmpeg_errors(full, capsys):
    stderr = f"{URL}: Server returned 401 Unauthorized\n".encode()
    code, _ = full(lambda cmd: subprocess.CompletedProcess(cmd, 1, b"", stderr))
    assert code == 1
    out = capsys.readouterr()
    assert "hunter2" not in out.out + out.err
    assert "rtsp://***@cam:554/stream1: Server returned 401 Unauthorized" in out.err


def test_full_timeout_does_not_echo_the_url(full, capsys):
    def hang(cmd):
        raise subprocess.TimeoutExpired(cmd, 30, stderr=f"{URL}: timed out".encode())

    code, _ = full(hang)
    assert code == 1
    out = capsys.readouterr()
    assert "hunter2" not in out.out + out.err


def test_capture_redaction_is_shared():
    from crib_monitor.capture import redact

    assert redact(f"x {URL} y") == "x rtsp://***@cam:554/stream1 y"
