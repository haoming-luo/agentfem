"""Shared publication must not expose a partial file after ordinary I/O failure."""

import errno

import pytest

from agentfem import _atomic_text


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("stage", ["write", "flush", "fsync"])
def test_partial_write_or_sync_failure_preserves_previous(tmp_path, monkeypatch, stage, existing):
    path = tmp_path / "published.json"
    previous = b'{"accepted": true}\n'
    if existing:
        path.write_bytes(previous)
    factory = _atomic_text.tempfile.NamedTemporaryFile

    class FailingStream:
        def __init__(self, stream):
            self.stream = stream
            self.name = stream.name

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return self.stream.__exit__(*args)

        def write(self, content):
            if stage == "write":
                self.stream.write(content[:5])
                raise OSError(errno.ENOSPC, "injected partial-write failure")
            return self.stream.write(content)

        def flush(self):
            self.stream.flush()
            if stage == "flush":
                raise OSError(errno.EIO, "injected flush failure")

        def fileno(self):
            return self.stream.fileno()

    monkeypatch.setattr(
        _atomic_text.tempfile, "NamedTemporaryFile",
        lambda **kwargs: FailingStream(factory(**kwargs)),
    )
    if stage == "fsync":
        def fail_sync(fd):
            raise OSError(errno.EIO, "injected fsync failure")
        monkeypatch.setattr(_atomic_text.os, "fsync", fail_sync)

    with pytest.raises(OSError, match="injected"):
        _atomic_text.atomic_write_text(path, '{"new": "not published"}\n')
    if existing:
        assert path.read_bytes() == previous
    else:
        assert not path.exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_success_publishes_utf8_and_returns_destination(tmp_path):
    path = tmp_path / "nested/result.json"
    assert _atomic_text.atomic_write_text(path, '{"label": "温度"}\n') == path
    assert path.read_text(encoding="utf-8") == '{"label": "温度"}\n'
    assert list(path.parent.iterdir()) == [path]
