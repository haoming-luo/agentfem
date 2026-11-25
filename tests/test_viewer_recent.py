# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
from agentfem._viewer_recent import local_path, recent_locations


def test_pasted_unicode_folder_and_shell_continuation(tmp_path):
    folder = tmp_path/'结果 with spaces'
    folder.mkdir()
    assert local_path(f' "{folder}" ') == folder
    assert local_path(str(folder)+'/\\') == folder


def test_recent_is_bounded_deduplicated_and_ignores_missing(tmp_path):
    store = tmp_path/'recent.json'
    for i in range(10):
        folder = tmp_path/str(i)
        folder.mkdir()
        recent_locations(store, folder)
    entries = recent_locations(store, tmp_path/'5')
    assert len(entries) == 8 and entries[0]['value'] == str(tmp_path/'5')
    (tmp_path/'5').rmdir()
    assert str(tmp_path/'5') not in [i['value'] for i in recent_locations(store)]
