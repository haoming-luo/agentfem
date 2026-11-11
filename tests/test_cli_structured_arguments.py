"""Machine-readable argument errors without solving or starting MPI."""

import json

import pytest

from agentfem import cli


@pytest.mark.parametrize("arguments", [
    ["check", "--json", "--does-not-exist"],
    ["unknown-command", "--json"],
    ["run", "--json", "--mpi", "not-an-integer"],
    ["check", "--json", "--project"],
    ["run", "--json", "--mpi", "0"],
    ["runs", "--json", "--limit", "-1"],
])
def test_invalid_json_request_is_one_structured_error(arguments, monkeypatch, capsys):
    class NoRuntime:
        def __getattr__(self, name):
            pytest.fail("Argument failure must not access MPI")

    monkeypatch.setattr(cli, "MPI", NoRuntime())
    assert cli.main(arguments) == 2
    output = capsys.readouterr()
    assert output.err == ""
    record = json.loads(output.out)
    assert record["status"] == "failed"
    assert record["stage"] == "argument_parsing"
    assert record["error"]["code"] == "AFM-CLI-ARGUMENT"


@pytest.mark.parametrize("arguments", [
    ["check", "--does-not-exist"],
    ["check", "--", "--json"],
])
def test_human_errors_and_case_arguments_preserve_argparse_behavior(arguments, capsys):
    with pytest.raises(SystemExit) as caught:
        cli.main(arguments)
    assert caught.value.code == 2
    output = capsys.readouterr()
    assert output.out == ""
    assert "usage:" in output.err
