"""Windows optional DLL search remains within the running interpreter."""
import os
from pathlib import Path
import sys
from unittest.mock import Mock

import pytest

from fpvsesh import runtime_dlls


def test_bracketed_library_path_is_literal_process_local_and_idempotent(tmp_path, monkeypatch):
    prefix = tmp_path / "optional [flight]"
    library = prefix / "Lib/site-packages/torch/lib"
    library.mkdir(parents=True)
    for name in ("torch_cpu.dll", "torch_python.dll"):
        (library / name).write_bytes(b"synthetic library placeholder")
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(sys, "prefix", str(prefix))
    monkeypatch.setattr(runtime_dlls, "_handles", {})
    add_directory = Mock(return_value=object())
    monkeypatch.setattr(os, "add_dll_directory", add_directory, raising=False)
    monkeypatch.setenv("PATH", "existing-path")
    monkeypatch.setenv("FPV_DLL_TEST_UNRELATED", "unchanged")
    runtime_dlls.prepare_torch_dlls()
    runtime_dlls.prepare_torch_dlls()
    add_directory.assert_called_once_with(str(library.resolve()))
    assert os.environ["PATH"].split(os.pathsep) == [str(library.resolve()), "existing-path"]
    assert os.environ["FPV_DLL_TEST_UNRELATED"] == "unchanged"


def test_resolved_library_outside_interpreter_is_rejected_before_environment_change(tmp_path, monkeypatch):
    prefix = tmp_path / "runtime"
    prefix.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    real_resolve = Path.resolve
    def resolve(path, *args, **kwargs):
        if path == prefix / "Lib/site-packages/torch/lib":
            return outside
        return real_resolve(path, *args, **kwargs)
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(sys, "prefix", str(prefix))
    monkeypatch.setattr(Path, "resolve", resolve)
    monkeypatch.setenv("PATH", "unchanged-path")
    add_directory = Mock()
    monkeypatch.setattr(os, "add_dll_directory", add_directory, raising=False)
    with pytest.raises(RuntimeError, match="inside this Python environment"):
        runtime_dlls.prepare_torch_dlls()
    add_directory.assert_not_called()
    assert os.environ["PATH"] == "unchanged-path"


def test_incomplete_libraries_do_not_change_search_path(tmp_path, monkeypatch):
    (tmp_path / "Lib/site-packages/torch/lib").mkdir(parents=True)
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(sys, "prefix", str(tmp_path))
    monkeypatch.setenv("PATH", "unchanged-path")
    with pytest.raises(RuntimeError, match="incomplete"):
        runtime_dlls.prepare_torch_dlls()
    assert os.environ["PATH"] == "unchanged-path"
