import shlex

import pytest

from agentfem._viewer_shortcut import create_shortcut


def test_mac_shortcut_is_bound_detached_and_quoted(tmp_path):
    location = tmp_path / "results 'with spaces'"
    location.mkdir()
    path = create_shortcut(tmp_path / "Viewer.command", location,
                           executable="/opt/my env/bin/python", platform="darwin")
    text = path.read_text()
    assert shlex.quote(str(location)) in text
    assert "'/opt/my env/bin/python'" in text
    assert "nohup" in text and '2>&1 </dev/null &' in text
    assert "mktemp" in text and "kill -0" in text
    assert path.stat().st_mode & 0o777 == 0o700
    with pytest.raises(FileExistsError):
        create_shortcut(path, location, platform="darwin")


def test_linux_shortcut_escapes_desktop_field_codes(tmp_path):
    location = tmp_path / "test%f"
    location.mkdir()
    path = create_shortcut(tmp_path / "Viewer.desktop", location,
                           executable="/opt/python", platform="linux")
    assert "test%%f" in path.read_text()
    assert 'Exec="/opt/python" "-m" "agentfem.viewer"' in path.read_text()


@pytest.mark.parametrize("platform,name", [("win32", "Viewer.cmd"), ("darwin", "Viewer.txt")])
def test_shortcut_rejects_unsupported_target(tmp_path, platform, name):
    with pytest.raises(ValueError):
        create_shortcut(tmp_path / name, tmp_path, platform=platform)
    assert not (tmp_path / name).exists()


def test_shortcut_rejects_newline_in_path(tmp_path):
    location = tmp_path / "line\nbreak"
    location.mkdir()
    with pytest.raises(ValueError, match="line breaks"):
        create_shortcut(tmp_path / "Viewer.desktop", location, platform="linux")
