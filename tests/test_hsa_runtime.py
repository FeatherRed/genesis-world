"""Unit tests for the ROCm HSA-runtime detection/repair helper ``genesis/utils/hsa.py``.

``genesis.utils.hsa`` exists because PyTorch sometimes bundles a ``libhsa-runtime64.so`` that corrupts AQL packets on
long ROCm rollouts, which is repaired by re-executing the interpreter once with the system runtime in ``LD_PRELOAD``.
Both halves are pure decision logic wrapped around the filesystem and the environment, so they are exercised here
without a GPU, without network access, and without ever starting a child process:

- ``inspect_hsa_runtime`` / ``system_hsa_runtime_path`` are driven against a *fake* torch installation built in
  ``tmp_path`` (a stub ``torch`` module whose ``__file__`` points at it) and a *fake* ROCm library directory that is
  injected through ``hsa._candidate_lib_dirs``.  The real host has ``/opt/rocm``, so that hook is always replaced:
  nothing here reads the machine's actual runtime.
- ``ensure_system_hsa_runtime`` is driven with a hand-built :class:`HsaRuntimeStatus` and a recording logger, while
  ``os.execve`` is replaced by a recorder.  The recorder *returns* instead of raising so that a stray re-exec shows up
  as a clean ``calls == []`` assertion rather than being swallowed by the helper's own broad ``except``.

The embedded binary version is exercised for real: the bundled library is named ``libhsa-runtime64.so`` (no version
suffix), so the module can only recover its version from the text embedded in the first megabytes of the file, which
is exactly what the fixture content mirrors.  ``sys.platform`` is patched per test because the module reads it at call
time.  There are no sleeps and no timing assumptions anywhere in this file.
"""

import hashlib
import os
import sys
import types
from pathlib import Path

import pytest

from genesis.constants import backend as gs_backend
from genesis.utils import hsa

pytestmark = [pytest.mark.required, pytest.mark.parametrize("backend", [None])]  # Disable genesis initialization

#: Version embedded in the bundled runtime, and the version carried by the fake system runtime's filename.
BUNDLED_VERSION = "1.18.70101"
SYSTEM_VERSION = "1.18.70100"

#: Fake binaries. The bundled one embeds its version (its filename does not carry one), the system one must simply
#: differ byte-wise so that its sha256 does not match torch's.
BUNDLED_CONTENT = (
    b"\x7fELF\x02\x01\x01" + b"\x00" * 32 + f"libhsa-runtime64.so.{BUNDLED_VERSION}".encode() + b"\x00" * 32
)
SYSTEM_CONTENT = (
    b"\x7fELF\x02\x01\x01" + b"\x01" * 32 + f"libhsa-runtime64.so.{SYSTEM_VERSION}".encode() + b"\x00" * 32
)

#: Every reason ``_warn_only_context`` can document; the repair helper must refuse to exec in each of them.
WARN_ONLY_REASONS = [
    "running under pytest",
    "running in an interactive interpreter",
    "launched via 'python -m'",
    "running without a real script path in sys.argv[0]",
    "running inside IPython/ipykernel",
]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _install_fake_torch(monkeypatch, package: Path, *, gpu_initialized: bool = False):
    """Put a stub ``torch`` module in ``sys.modules`` whose ``__file__`` points into the fake installation."""
    module = types.ModuleType("torch")
    module.__file__ = str(package / "__init__.py")
    module.cuda = types.SimpleNamespace(is_initialized=lambda: gpu_initialized)
    monkeypatch.setitem(sys.modules, "torch", module)
    return module


def _install_system_runtime(monkeypatch, lib_dir: Path, name: str = f"libhsa-runtime64.so.{SYSTEM_VERSION}",
                            content: bytes = SYSTEM_CONTENT) -> Path:
    """Create a fake ROCm library directory and make it the only one ``_candidate_lib_dirs`` reports."""
    lib_dir.mkdir(parents=True, exist_ok=True)
    system_path = lib_dir / name
    system_path.write_bytes(content)
    monkeypatch.setattr(hsa, "_candidate_lib_dirs", lambda: [lib_dir])
    return system_path


def _avoid_real_rocm_dirs(monkeypatch, lib_dir: Path) -> None:
    """Point the candidate search at a single (possibly missing) directory, away from the host's ``/opt/rocm``."""
    monkeypatch.setattr(hsa, "_candidate_lib_dirs", lambda: [lib_dir])


@pytest.fixture
def hsa_env(monkeypatch):
    """Neutral process state: Linux, no HSA re-exec flags, no preload, and no host ROCm directories."""
    for name in ("GS_HSA_REEXEC", "GS_HSA_REEXEC_DONE", "LD_PRELOAD", "ROCM_PATH", "ROCM_HOME"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(sys, "platform", "linux")
    # The host really has /opt/rocm; these unit tests must never see it.
    monkeypatch.setattr(hsa, "_candidate_lib_dirs", list)


@pytest.fixture
def torch_install(tmp_path, monkeypatch, hsa_env):
    """A fake ROCm PyTorch installation plus a candidate search that finds nothing."""
    root = tmp_path / "site-packages"
    package = root / "torch"
    package.mkdir(parents=True)
    (package / "__init__.py").write_bytes(b"")
    lib = package / "lib"
    lib.mkdir()
    bundled = lib / "libhsa-runtime64.so"
    bundled.write_bytes(BUNDLED_CONTENT)
    (lib / "libamdhip64.so").write_bytes(b"")
    module = _install_fake_torch(monkeypatch, package)
    _avoid_real_rocm_dirs(monkeypatch, tmp_path / "no-such-rocm" / "lib")
    return types.SimpleNamespace(root=root, package=package, lib=lib, bundled=bundled, module=module)


# ---------------------------------------------------------------------------------------------------------------------
# Detection: inspect_hsa_runtime / system_hsa_runtime_path
# ---------------------------------------------------------------------------------------------------------------------


def test_inspect_not_applicable_off_linux(monkeypatch, torch_install):
    monkeypatch.setattr(sys, "platform", "darwin")

    status = hsa.inspect_hsa_runtime()

    assert status.applicable is False
    assert status.needs_reexec is False
    assert "Linux" in status.reason
    assert hsa.system_hsa_runtime_path() is None


def test_inspect_not_applicable_without_torch(monkeypatch, hsa_env):
    monkeypatch.setattr(hsa, "_torch_lib_dir", lambda: None)

    status = hsa.inspect_hsa_runtime()

    assert status.applicable is False
    assert status.needs_reexec is False
    assert "could not be located" in status.reason
    assert hsa.system_hsa_runtime_path() is None


@pytest.mark.parametrize("missing", ["libhsa-runtime64.so", "libamdhip64.so"])
def test_inspect_not_applicable_without_a_bundled_rocm_runtime(torch_install, missing):
    (torch_install.lib / missing).unlink()

    status = hsa.inspect_hsa_runtime()

    assert status.applicable is False
    assert status.needs_reexec is False
    assert "does not bundle" in status.reason
    assert hsa.system_hsa_runtime_path() is None


@pytest.mark.parametrize("create_dir", [True, False], ids=["empty-lib-dir", "missing-lib-dir"])
def test_inspect_without_system_candidate(torch_install, monkeypatch, tmp_path, create_dir):
    lib_dir = tmp_path / "rocm" / "lib"
    if create_dir:
        lib_dir.mkdir(parents=True)
    _avoid_real_rocm_dirs(monkeypatch, lib_dir)

    status = hsa.inspect_hsa_runtime()

    assert status.applicable is True
    assert status.bundled_path == torch_install.bundled
    assert status.system_path is None
    assert status.system_version is None
    assert status.system_sha256 is None
    assert status.needs_reexec is False
    assert status.blocked_reason is None
    assert "No system HSA runtime differing" in status.reason
    assert hsa.system_hsa_runtime_path() is None


def test_inspect_matching_system_candidate_needs_no_reexec(torch_install, monkeypatch, tmp_path):
    system_path = _install_system_runtime(monkeypatch, tmp_path / "rocm" / "lib", content=BUNDLED_CONTENT)
    assert system_path.read_bytes() == torch_install.bundled.read_bytes()

    status = hsa.inspect_hsa_runtime()

    assert status.applicable is True
    assert status.system_path is None
    assert status.system_sha256 is None
    assert status.needs_reexec is False
    assert status.blocked_reason is None
    assert "No system HSA runtime differing" in status.reason
    assert hsa.system_hsa_runtime_path() is None


def test_inspect_mismatching_candidate_requires_reexec(torch_install, monkeypatch, tmp_path):
    system_path = _install_system_runtime(monkeypatch, tmp_path / "rocm" / "lib")

    status = hsa.inspect_hsa_runtime()

    assert status.applicable is True
    assert status.bundled_path == torch_install.bundled
    # The bundled library has no version suffix, so its version comes from the embedded binary text.
    assert status.bundled_version == BUNDLED_VERSION
    assert status.bundled_sha256 == _sha256(BUNDLED_CONTENT)
    assert status.system_path == system_path
    assert status.system_version == SYSTEM_VERSION
    assert status.system_sha256 == _sha256(SYSTEM_CONTENT)
    assert status.system_sha256 != status.bundled_sha256
    assert status.preload is None
    assert status.preload_names_system is False
    assert status.needs_reexec is True
    assert status.blocked_reason is None
    assert "a re-exec preloading the system runtime is required" in status.reason
    assert hsa.system_hsa_runtime_path() == system_path


@pytest.mark.parametrize(
    "variant",
    ["exact-path", "same-bytes-other-name", "colon-and-space-separated", "unrelated"],
)
def test_inspect_preload_names_system(torch_install, monkeypatch, tmp_path, variant):
    system_path = _install_system_runtime(monkeypatch, tmp_path / "rocm" / "lib")
    copy_path = tmp_path / "preload" / "libhsa-runtime64.so"
    copy_path.parent.mkdir()
    copy_path.write_bytes(SYSTEM_CONTENT)
    unrelated_path = tmp_path / "vendor" / "libunnamed.so"
    unrelated_path.parent.mkdir()
    unrelated_path.write_bytes(b"not the system runtime")

    if variant == "exact-path":
        preload, expected = str(system_path), True
    elif variant == "same-bytes-other-name":
        preload, expected = str(copy_path), True
    elif variant == "colon-and-space-separated":
        preload, expected = f"/opt/vendor/libdeep.so {copy_path}", True
    else:
        preload, expected = str(unrelated_path), False
    monkeypatch.setenv("LD_PRELOAD", preload)

    status = hsa.inspect_hsa_runtime()

    assert status.preload == preload
    assert status.preload_names_system is expected
    assert status.needs_reexec is (not expected)
    assert ("already preloaded" in status.reason) is expected
    assert hsa._preload_names_path(preload, system_path, status.system_sha256) is expected


def test_inspect_sentinel_disables_reexec(torch_install, monkeypatch, tmp_path):
    system_path = _install_system_runtime(monkeypatch, tmp_path / "rocm" / "lib")
    monkeypatch.setenv("GS_HSA_REEXEC_DONE", "1")

    status = hsa.inspect_hsa_runtime()

    assert status.applicable is True
    assert status.system_path == system_path
    assert status.needs_reexec is False
    assert status.blocked_reason is not None
    assert "GS_HSA_REEXEC_DONE=1" in status.blocked_reason
    assert "already happened" in status.reason


def test_inspect_opt_out_blocks_but_keeps_the_technical_need(torch_install, monkeypatch, tmp_path):
    system_path = _install_system_runtime(monkeypatch, tmp_path / "rocm" / "lib")
    monkeypatch.setenv("GS_HSA_REEXEC", "0")

    status = hsa.inspect_hsa_runtime()

    assert status.applicable is True
    assert status.system_path == system_path
    # The opt-out only blocks the repair; it does not pretend the re-exec is unnecessary.
    assert status.needs_reexec is True
    assert status.blocked_reason is not None
    assert "GS_HSA_REEXEC=0" in status.blocked_reason
    assert "a re-exec preloading the system runtime is required" in status.reason


def test_inspect_never_raises(hsa_env, monkeypatch):
    def boom():
        raise RuntimeError("boom")

    monkeypatch.setattr(hsa, "_torch_lib_dir", boom)

    status = hsa.inspect_hsa_runtime()

    assert status.applicable is False
    assert status.needs_reexec is False
    assert "failed unexpectedly" in status.reason
    assert "boom" in status.reason


# ---------------------------------------------------------------------------------------------------------------------
# Private helpers, tested directly where that is cleaner than going through the status object
# ---------------------------------------------------------------------------------------------------------------------


def test_torch_lib_dir_from_a_loaded_module(hsa_env, monkeypatch, tmp_path):
    package = tmp_path / "torch"
    package.mkdir()
    (package / "__init__.py").write_bytes(b"")
    _install_fake_torch(monkeypatch, package)

    assert hsa._torch_lib_dir() == package / "lib"


def test_torch_lib_dir_without_a_module_file(hsa_env, monkeypatch, tmp_path):
    package = tmp_path / "torch"
    package.mkdir()
    (package / "__init__.py").write_bytes(b"")
    module = _install_fake_torch(monkeypatch, package)
    module.__file__ = None

    assert hsa._torch_lib_dir() is None


def test_torch_lib_dir_from_find_spec(hsa_env, monkeypatch, tmp_path):
    origin = tmp_path / "site-packages" / "torch" / "__init__.py"
    origin.parent.mkdir(parents=True)
    origin.write_bytes(b"")
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    # torch is never imported for this: the location is taken from the module spec.
    spec = types.SimpleNamespace(origin=str(origin))
    monkeypatch.setattr(hsa, "importlib", _fake_importlib(lambda name: spec))

    assert hsa._torch_lib_dir() == origin.parent / "lib"


def _fake_importlib(find_spec):
    """Stand-in for the ``importlib`` module used by ``hsa``, so that torch is never really located."""
    return types.SimpleNamespace(util=types.SimpleNamespace(find_spec=find_spec))


def _find_spec_returning_none(name):
    return None


def _find_spec_failing(error):
    def find_spec(name):
        raise error

    return find_spec


@pytest.mark.parametrize(
    "find_spec",
    [
        _find_spec_returning_none,
        _find_spec_failing(ValueError("torch.__spec__ is None")),
        _find_spec_failing(ImportError("no torch")),
    ],
    ids=["no-spec", "value-error", "import-error"],
)
def test_torch_lib_dir_is_none_when_torch_cannot_be_located(hsa_env, monkeypatch, find_spec):
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.setattr(hsa, "importlib", _fake_importlib(find_spec))

    assert hsa._torch_lib_dir() is None


def test_candidate_lib_dirs_prioritizes_environment_roots(tmp_path, monkeypatch):
    root = tmp_path / "rocm"
    # Both variables naming the same root must not duplicate the search entries.
    monkeypatch.setenv("ROCM_PATH", str(root))
    monkeypatch.setenv("ROCM_HOME", str(root))

    lib_dirs = hsa._candidate_lib_dirs()

    assert lib_dirs[:2] == [root / "lib", root / "lib64"]
    real_paths = [os.path.realpath(lib_dir) for lib_dir in lib_dirs]
    assert len(real_paths) == len(set(real_paths))
    # The default ROCm root is always appended, after the environment-provided ones.
    assert Path("/opt/rocm") / "lib" in lib_dirs


@pytest.mark.parametrize("preload", [None, "", "   ", ":"])
def test_preload_names_path_is_false_without_entries(tmp_path, preload):
    system_path = tmp_path / "libhsa-runtime64.so.1"
    system_path.write_bytes(SYSTEM_CONTENT)

    assert hsa._preload_names_path(preload, system_path, _sha256(SYSTEM_CONTENT)) is False


def test_preload_names_path_matches_by_path_or_by_content(tmp_path):
    system_path = tmp_path / "rocm" / "lib" / "libhsa-runtime64.so.1"
    system_path.parent.mkdir(parents=True)
    system_path.write_bytes(SYSTEM_CONTENT)
    alias_path = tmp_path / "alias" / "libhsa-runtime64.so"
    alias_path.parent.mkdir()
    alias_path.write_bytes(SYSTEM_CONTENT)
    other_path = tmp_path / "other.so"
    other_path.write_bytes(b"something else")

    assert hsa._preload_names_path(str(system_path), system_path, None) is True
    assert hsa._preload_names_path(f"/opt/vendor/libdeep.so:{system_path}", system_path, None) is True
    assert hsa._preload_names_path(str(alias_path), system_path, _sha256(SYSTEM_CONTENT)) is True
    assert hsa._preload_names_path(str(other_path), system_path, _sha256(SYSTEM_CONTENT)) is False
    # A missing entry cannot be hashed, so it never matches.
    assert hsa._preload_names_path(str(tmp_path / "missing.so"), system_path, _sha256(SYSTEM_CONTENT)) is False


def test_scan_embedded_version(tmp_path):
    bundled = tmp_path / "libhsa-runtime64.so"
    bundled.write_bytes(BUNDLED_CONTENT)
    no_version = tmp_path / "libhsa-runtime64-noversion.so"
    no_version.write_bytes(b"\x00" * 64)

    assert hsa._scan_embedded_version(bundled) == BUNDLED_VERSION
    assert hsa._scan_embedded_version(no_version) is None
    assert hsa._scan_embedded_version(tmp_path / "missing.so") is None


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("libhsa-runtime64.so", None),
        ("libhsa-runtime64.so.1", "1"),
        (f"libhsa-runtime64.so.{SYSTEM_VERSION}", SYSTEM_VERSION),
        ("libhsa-runtime64.so.noversion", None),
        ("libamdhip64.so.1", None),
    ],
)
def test_version_from_filename(filename, expected):
    assert hsa._version_from_filename(Path("/fake") / filename) == expected


@pytest.mark.parametrize("initialized", [True, False])
def test_torch_gpu_initialized(hsa_env, monkeypatch, tmp_path, initialized):
    package = tmp_path / "torch"
    package.mkdir()
    (package / "__init__.py").write_bytes(b"")
    _install_fake_torch(monkeypatch, package, gpu_initialized=initialized)

    assert hsa._torch_gpu_initialized() is initialized


def test_torch_gpu_initialized_without_torch(hsa_env, monkeypatch):
    monkeypatch.delitem(sys.modules, "torch", raising=False)

    assert hsa._torch_gpu_initialized() is False


def test_torch_gpu_initialized_survives_torch_errors(hsa_env, monkeypatch, tmp_path):
    package = tmp_path / "torch"
    package.mkdir()
    (package / "__init__.py").write_bytes(b"")
    module = _install_fake_torch(monkeypatch, package)

    def boom():
        raise RuntimeError("cuda is unhappy")

    module.cuda = types.SimpleNamespace(is_initialized=boom)

    assert hsa._torch_gpu_initialized() is False


# ---------------------------------------------------------------------------------------------------------------------
# _warn_only_context: why a re-exec would be unsafe
# ---------------------------------------------------------------------------------------------------------------------


def _prepare_reexec_safe_context(monkeypatch, tmp_path) -> Path:
    """Put the interpreter in the state ``_warn_only_context`` considers safe for a re-exec."""
    monkeypatch.delenv("PYTEST_VERSION", raising=False)
    # pytest is imported for the whole run and the gate only looks it up in sys.modules, so drop it here; monkeypatch
    # restores it (and the real ``__main__`` spec / argv / flags) at teardown.
    for module_name in ("pytest", "IPython", "ipykernel"):
        monkeypatch.delitem(sys.modules, module_name, raising=False)
    monkeypatch.setattr(sys, "flags", types.SimpleNamespace(interactive=0))
    monkeypatch.setattr(sys.modules["__main__"], "__spec__", None)
    script_path = tmp_path / "script.py"
    script_path.write_text("print('hello')\n")
    monkeypatch.setattr(sys, "argv", [str(script_path)])
    return script_path


def test_warn_only_context_is_none_for_a_plain_script(monkeypatch, tmp_path):
    script_path = _prepare_reexec_safe_context(monkeypatch, tmp_path)

    assert script_path.is_file()
    assert hsa._warn_only_context() is None


def test_warn_only_context_under_pytest(monkeypatch, tmp_path):
    _prepare_reexec_safe_context(monkeypatch, tmp_path)
    monkeypatch.setenv("PYTEST_VERSION", "8.4.0")

    assert hsa._warn_only_context() == "running under pytest"


def test_warn_only_context_under_pytest_without_the_env_var(monkeypatch, tmp_path):
    _prepare_reexec_safe_context(monkeypatch, tmp_path)
    monkeypatch.setitem(sys.modules, "pytest", types.ModuleType("pytest"))

    assert hsa._warn_only_context() == "running under pytest"


def test_warn_only_context_in_the_natural_pytest_state():
    # No monkeypatching at all: this process really is pytest, which must win over everything else.
    assert hsa._warn_only_context() == "running under pytest"


def test_warn_only_context_in_an_interactive_interpreter(monkeypatch, tmp_path):
    _prepare_reexec_safe_context(monkeypatch, tmp_path)
    monkeypatch.setattr(sys, "flags", types.SimpleNamespace(interactive=1))

    assert hsa._warn_only_context() == "running in an interactive interpreter"


def test_warn_only_context_launched_with_python_dash_m(monkeypatch, tmp_path):
    _prepare_reexec_safe_context(monkeypatch, tmp_path)
    monkeypatch.setattr(sys.modules["__main__"], "__spec__", types.SimpleNamespace(name="pytest"))

    assert hsa._warn_only_context() == "launched via 'python -m'"


def test_warn_only_context_without_a_real_script_path(monkeypatch, tmp_path):
    _prepare_reexec_safe_context(monkeypatch, tmp_path)
    monkeypatch.setattr(sys, "argv", ["/nonexistent/xyz"])

    reason = hsa._warn_only_context()

    assert reason is not None
    assert "argv[0]" in reason


@pytest.mark.parametrize("module_name", ["IPython", "ipykernel"])
def test_warn_only_context_inside_ipython(monkeypatch, tmp_path, module_name):
    _prepare_reexec_safe_context(monkeypatch, tmp_path)
    monkeypatch.setitem(sys.modules, module_name, types.ModuleType(module_name))

    assert hsa._warn_only_context() == "running inside IPython/ipykernel"


# ---------------------------------------------------------------------------------------------------------------------
# ensure_system_hsa_runtime: every gate, plus the single re-exec path
# ---------------------------------------------------------------------------------------------------------------------


class RecordingLogger:
    """Minimal stand-in for the ``logger`` argument, collecting the emitted lines verbatim."""

    def __init__(self):
        self.infos = []
        self.warnings = []

    def info(self, message):
        self.infos.append(message)

    def warning(self, message):
        self.warnings.append(message)

    def __repr__(self):
        return f"RecordingLogger(infos={self.infos}, warnings={self.warnings})"


def _record_exec(monkeypatch):
    """Replace ``os.execve`` with a recorder; a stray call is caught by ``assert calls == []`` in the caller."""
    calls = []

    def _execve(path, argv, env):
        calls.append((path, list(argv), dict(env)))
        return None

    monkeypatch.setattr(hsa.os, "execve", _execve)
    return calls


def _patch_status(monkeypatch, status):
    monkeypatch.setattr(hsa, "inspect_hsa_runtime", lambda: status)


def _patch_reexec_allowed(monkeypatch):
    monkeypatch.setattr(hsa, "_warn_only_context", lambda: None)
    monkeypatch.setattr(hsa, "_torch_gpu_initialized", lambda: False)


def _make_status(tmp_path, **overrides):
    """A fully consistent status describing a differing, not-yet-preloaded system runtime."""
    bundled_path = tmp_path / "torch" / "lib" / "libhsa-runtime64.so"
    bundled_path.parent.mkdir(parents=True, exist_ok=True)
    bundled_path.write_bytes(BUNDLED_CONTENT)
    system_path = tmp_path / "rocm" / "lib" / f"libhsa-runtime64.so.{SYSTEM_VERSION}"
    system_path.parent.mkdir(parents=True, exist_ok=True)
    system_path.write_bytes(SYSTEM_CONTENT)
    values = {
        "applicable": True,
        "needs_reexec": True,
        "reason": "PyTorch bundles a differing HSA runtime, so a re-exec preloading the system runtime is required.",
        "bundled_path": bundled_path,
        "bundled_version": BUNDLED_VERSION,
        "bundled_sha256": _sha256(BUNDLED_CONTENT),
        "system_path": system_path,
        "system_version": SYSTEM_VERSION,
        "system_sha256": _sha256(SYSTEM_CONTENT),
        "preload": None,
        "preload_names_system": False,
        "blocked_reason": None,
    }
    values.update(overrides)
    return hsa.HsaRuntimeStatus(**values)


@pytest.mark.parametrize("requested_backend", [gs_backend.cpu, gs_backend.metal], ids=["cpu", "metal"])
def test_ensure_skips_cpu_and_metal_silently(tmp_path, monkeypatch, hsa_env, requested_backend):
    status = _make_status(tmp_path)
    _patch_status(monkeypatch, status)
    calls = _record_exec(monkeypatch)
    logger = RecordingLogger()

    assert hsa.ensure_system_hsa_runtime(requested_backend=requested_backend, logger=logger) is False
    assert calls == []
    assert logger.infos == []
    assert logger.warnings == []


def test_ensure_warns_on_opt_out(tmp_path, monkeypatch, hsa_env):
    status = _make_status(tmp_path)
    _patch_status(monkeypatch, status)
    monkeypatch.setenv("GS_HSA_REEXEC", "0")
    calls = _record_exec(monkeypatch)
    logger = RecordingLogger()

    assert hsa.ensure_system_hsa_runtime(requested_backend=gs_backend.gpu, logger=logger) is False
    assert calls == []
    assert logger.infos == []
    assert len(logger.warnings) == 1
    warning = logger.warnings[0]
    assert "GS_HSA_REEXEC=0" in warning
    assert "will be used instead of the system one" in warning
    assert str(status.bundled_path) in warning
    assert str(status.system_path) in warning


def test_ensure_is_silent_after_a_reexec(tmp_path, monkeypatch, hsa_env):
    status = _make_status(tmp_path)
    _patch_status(monkeypatch, status)
    monkeypatch.setenv("GS_HSA_REEXEC_DONE", "1")
    calls = _record_exec(monkeypatch)
    logger = RecordingLogger()

    assert hsa.ensure_system_hsa_runtime(requested_backend=gs_backend.gpu, logger=logger) is False
    assert calls == []
    assert logger.infos == []
    assert logger.warnings == []


@pytest.mark.parametrize(
    "overrides",
    [
        {"applicable": False, "needs_reexec": False, "system_path": None, "reason": "not applicable"},
        {"system_path": None, "needs_reexec": False},
        {"preload_names_system": True, "needs_reexec": False},
    ],
    ids=["not-applicable", "no-system-path", "already-preloaded"],
)
def test_ensure_skips_without_a_differing_unloaded_system_runtime(tmp_path, monkeypatch, hsa_env, overrides):
    status = _make_status(tmp_path, **overrides)
    _patch_status(monkeypatch, status)
    calls = _record_exec(monkeypatch)
    logger = RecordingLogger()

    assert hsa.ensure_system_hsa_runtime(requested_backend=gs_backend.gpu, logger=logger) is False
    assert calls == []
    assert logger.infos == []
    assert logger.warnings == []


@pytest.mark.parametrize("reason", WARN_ONLY_REASONS)
def test_ensure_warns_when_the_process_is_not_reexec_safe(tmp_path, monkeypatch, hsa_env, reason):
    status = _make_status(tmp_path)
    _patch_status(monkeypatch, status)
    monkeypatch.setattr(hsa, "_warn_only_context", lambda: reason)
    monkeypatch.setattr(hsa, "_torch_gpu_initialized", lambda: False)
    calls = _record_exec(monkeypatch)
    logger = RecordingLogger()

    assert hsa.ensure_system_hsa_runtime(requested_backend=gs_backend.gpu, logger=logger) is False
    assert calls == []
    assert logger.infos == []
    assert len(logger.warnings) == 1
    assert "Skipping the automatic HSA runtime re-exec" in logger.warnings[0]
    assert reason in logger.warnings[0]


def test_ensure_warns_when_torch_gpu_is_already_initialized(tmp_path, monkeypatch, hsa_env):
    status = _make_status(tmp_path)
    _patch_status(monkeypatch, status)
    monkeypatch.setattr(hsa, "_warn_only_context", lambda: None)
    monkeypatch.setattr(hsa, "_torch_gpu_initialized", lambda: True)
    calls = _record_exec(monkeypatch)
    logger = RecordingLogger()

    assert hsa.ensure_system_hsa_runtime(requested_backend=gs_backend.gpu, logger=logger) is False
    assert calls == []
    assert logger.infos == []
    assert len(logger.warnings) == 1
    assert "already initialized" in logger.warnings[0]


def test_ensure_reexecs_with_the_system_runtime_preloaded(tmp_path, monkeypatch, hsa_env):
    status = _make_status(tmp_path)
    _patch_status(monkeypatch, status)
    _patch_reexec_allowed(monkeypatch)
    monkeypatch.setenv("LD_PRELOAD", "/opt/vendor/libdeep.so")
    calls = _record_exec(monkeypatch)
    logger = RecordingLogger()

    assert hsa.ensure_system_hsa_runtime(requested_backend=gs_backend.gpu, logger=logger) is False

    assert len(calls) == 1
    exec_path, exec_argv, exec_env = calls[0]
    assert exec_path == sys.executable
    assert exec_argv == [sys.executable, os.path.abspath(sys.argv[0]), *sys.argv[1:]]
    assert exec_env["GS_HSA_REEXEC_DONE"] == "1"
    assert exec_env["LD_PRELOAD"].startswith(str(status.system_path))
    assert exec_env["LD_PRELOAD"] == f"{status.system_path}:/opt/vendor/libdeep.so"
    # The parent environment is left untouched: only the child's copy carries the sentinel and the preload.
    assert os.environ.get("GS_HSA_REEXEC_DONE") is None
    assert os.environ["LD_PRELOAD"] == "/opt/vendor/libdeep.so"

    assert len(logger.infos) == 1
    assert "Re-executing the interpreter" in logger.infos[0]
    assert str(status.system_path) in logger.infos[0]
    assert logger.warnings == []


def test_ensure_warns_when_the_reexec_fails(tmp_path, monkeypatch, hsa_env):
    status = _make_status(tmp_path)
    _patch_status(monkeypatch, status)
    _patch_reexec_allowed(monkeypatch)
    attempts = []

    def _execve(path, argv, env):
        attempts.append(path)
        raise OSError(8, "Exec format error")

    monkeypatch.setattr(hsa.os, "execve", _execve)
    logger = RecordingLogger()

    assert hsa.ensure_system_hsa_runtime(requested_backend=gs_backend.gpu, logger=logger) is False

    assert attempts == [sys.executable]
    assert len(logger.warnings) == 1
    assert "Failed to re-execute the interpreter" in logger.warnings[0]
    assert str(status.system_path) in logger.warnings[0]
    assert "Exec format error" in logger.warnings[0]


def test_ensure_never_raises_when_inspection_fails(monkeypatch, hsa_env):
    def boom():
        raise RuntimeError("boom")

    monkeypatch.setattr(hsa, "inspect_hsa_runtime", boom)
    logger = RecordingLogger()

    assert hsa.ensure_system_hsa_runtime(requested_backend=gs_backend.gpu, logger=logger) is False
    assert len(logger.warnings) == 1
    assert "failed unexpectedly" in logger.warnings[0]
    assert "boom" in logger.warnings[0]
