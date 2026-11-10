"""Reusable \"PyTorch unavailable\" harness for the torch-free Genesis test surface.

Genesis must import and simulate where PyTorch is not installed, so the tests that prove it cannot import PyTorch
themselves: the parent pytest process has it, and the point of the exercise is that the *child* does not.  This
module is the single place where that child is built, so that every torch-free test starts from the same blocker,
the same environment and the same diagnostics:

- :data:`TORCH_BLOCK_SOURCE` is a ``sys.meta_path`` finder that refuses ``torch`` and every ``torch.*`` name.  It is
  prepended to the child's source, i.e. it is installed before anything can import PyTorch, which is what makes a
  leaked ``import torch`` deep inside the package surface as an :class:`ImportError` instead of silently working.
- :func:`run_python` runs one script in a child interpreter of the very same executable, blocks PyTorch by default,
  and appends an epilogue that fails the child if ``torch`` ended up in ``sys.modules`` anyway (a module injected by
  a C extension or a ``sys.modules`` write would bypass the finder otherwise).  :func:`run_python_checked` turns a
  non-zero exit code into an :class:`AssertionError` carrying the child's whole output, because a bare
  ``CalledProcessError`` in a pytest report hides the traceback that explains the failure.
- :func:`torch_import_violations` / :func:`assert_no_torch_imports` are the static half: an AST scan of the files on
  the torch-free training path that reports ``<path>:<lineno>: <statement>`` for every ``import torch``,
  ``import torch.x``, ``from torch import y`` and ``from torch.x import y``.
- :func:`has_gpu` answers the parents' \"can a GPU-backed Genesis run happen here\" question without importing
  torch or quadrants.

This module is deliberately *not* collected by pytest (no ``test_`` prefix); it is imported by
``tests/test_torch_free_training.py`` and can be reused by any future torch-free test::

    from tests.torch_block import run_python_checked

Nothing here starts a GPU run on its own: the scene scripts live in the test module that needs them.
"""

from __future__ import annotations

import ast
import glob
import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

__all__ = [
    "REPO_ROOT",
    "EXAMPLES_DIR",
    "GENESIS_DIR",
    "TORCH_BLOCK_SOURCE",
    "run_python",
    "run_python_checked",
    "torch_import_violations",
    "assert_no_torch_imports",
    "has_gpu",
]

#: Repository root, derived from this file so that the harness works in any worktree and from any cwd.
REPO_ROOT = Path(__file__).resolve().parents[1]
#: Root of the runnable demos, which the torch-free training path lives under.
EXAMPLES_DIR = REPO_ROOT / "examples"
#: The Genesis package, most of which must stay importable without PyTorch.
GENESIS_DIR = REPO_ROOT / "genesis"

#: Kernel device nodes that only exist on a machine with a GPU, used when no management library is installed.
GPU_DEVICE_NODES = ("/dev/nvidiactl", "/dev/kfd")

#: Source prepended to a torch-free child, installing the blocker before any other import.
#: A finder raising from 'find_spec' is what 'import torch' itself triggers, so the failure surfaces where the
#: offending import is written instead of later, at the first attribute access.
TORCH_BLOCK_SOURCE = '''\
import sys


class _BlockTorch:
    """Meta-path finder that makes every name of the PyTorch namespace unimportable."""

    def find_module(self, name, path=None):
        # Python 2-style protocol, still probed by a few libraries through 'importlib'
        return self.find_spec(name, path)

    def find_spec(self, name, path=None, target=None):
        if name == "torch" or name.startswith("torch."):
            raise ImportError(f"PyTorch is blocked by the torch-free test harness: '{name}'")
        return None


sys.meta_path.insert(0, _BlockTorch())
'''

#: Epilogue appended to a torch-free child: a finder is only consulted for a name that is not in 'sys.modules' yet,
#: so a module injected by an extension or by a plain assignment would slip through it undetected.
TORCH_ABSENT_EPILOGUE = '''\
_leaked = sorted(name for name in sys.modules if name == "torch" or name.startswith("torch."))
if _leaked:
    raise AssertionError(f"PyTorch leaked into the torch-free child process: {_leaked}")
'''

#: Directory handed to matplotlib through 'MPLCONFIGDIR', created on first use, never removed (it lives in '/tmp').
_MPL_CONFIG_DIR: str | None = None


def _mpl_config_dir() -> str:
    global _MPL_CONFIG_DIR
    if _MPL_CONFIG_DIR is None:
        _MPL_CONFIG_DIR = tempfile.mkdtemp(prefix="gs-torch-block-")
    return _MPL_CONFIG_DIR


def _child_source(script: str, *, block_torch: bool) -> str:
    body = textwrap.dedent(script).strip("\n")
    if not block_torch:
        return body + "\n"
    return "\n\n".join((TORCH_BLOCK_SOURCE, body, TORCH_ABSENT_EPILOGUE))


def run_python(
    script: str,
    *,
    block_torch: bool = True,
    timeout: float = 1200,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess:
    """Run ``script`` in a child interpreter, with PyTorch blocked unless ``block_torch`` is False.

    The child runs with ``cwd`` set to :data:`REPO_ROOT`, so ``import genesis`` resolves to this checkout, and it
    inherits the caller's environment (plus optional ``env`` overrides and a writable ``MPLCONFIGDIR``) because the
    GPU affinity and cache settings the test session exported must reach it.
    """
    child_env = dict(os.environ)
    if env is not None:
        child_env.update(env)
    # Matplotlib refuses to import when its default config directory is not writable, which happens in some sandboxes
    child_env.setdefault("MPLCONFIGDIR", _mpl_config_dir())
    return subprocess.run(
        [sys.executable, "-c", _child_source(script, block_torch=block_torch)],
        cwd=str(REPO_ROOT),
        env=child_env,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )


def run_python_checked(script: str, **kwargs) -> subprocess.CompletedProcess:
    """:func:`run_python`, raising an :class:`AssertionError` with the child's full output on a non-zero exit."""
    result = run_python(script, **kwargs)
    if result.returncode != 0:
        raise AssertionError(
            f"The child interpreter exited with code {result.returncode}.\n"
            f"--- child stdout ---\n{result.stdout}\n"
            f"--- child stderr ---\n{result.stderr}"
        )
    return result


def _as_path(entry) -> Path:
    """Resolve ``entry`` against :data:`REPO_ROOT` when it is not absolute."""
    path = Path(entry)
    return (path if path.is_absolute() else REPO_ROOT / path).resolve()


def _expand_entry(entry) -> list[Path]:
    """The existing paths ``entry`` designates, glob patterns expanded and missing paths dropped."""
    path = Path(entry)
    pattern = str(path if path.is_absolute() else REPO_ROOT / path)
    if any(char in pattern for char in "*?["):
        return sorted(Path(match) for match in glob.glob(pattern, recursive=True))
    resolved = _as_path(entry)
    return [resolved] if resolved.exists() else []


def _iter_python_files(paths, *, exclude=()) -> list[Path]:
    """Every ``.py`` file under ``paths``, deduplicated, sorted and pruned by ``exclude``.

    ``exclude`` entries are files or directories: a file matches by identity, a directory by being an ancestor of the
    candidate, which is how vendored trees are skipped without listing their modules.
    """
    excluded = tuple(_as_path(entry) for entry in exclude)
    files: dict[Path, None] = {}
    for entry in paths:
        for candidate in _expand_entry(entry):
            if candidate.is_dir():
                candidates = candidate.rglob("*.py")
            elif candidate.suffix == ".py":
                candidates = (candidate,)
            else:
                continue
            for file in candidates:
                resolved = file.resolve()
                if resolved in files:
                    continue
                if any(resolved == path or path in resolved.parents for path in excluded):
                    continue
                files[resolved] = None
    return sorted(files)


def _is_torch_name(name: str | None) -> bool:
    return name is not None and (name == "torch" or name.startswith("torch."))


def _torch_imports(path: Path) -> list[tuple[int, str]]:
    """The ``(lineno, statement)`` pairs of every PyTorch import in ``path``, in source order."""
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(_is_torch_name(alias.name) for alias in node.names):
                found.append((node.lineno, ast.unparse(node)))
        elif isinstance(node, ast.ImportFrom):
            # A relative import ('level' > 0) can never reach the installed PyTorch package
            if node.level == 0 and _is_torch_name(node.module):
                found.append((node.lineno, ast.unparse(node)))
    return sorted(found)


def _label(path: Path) -> str:
    """Print a scanned file relative to the repository root, which keeps the report stable across worktrees."""
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def torch_import_violations(paths, *, allowlist=(), exclude=()) -> list[str]:
    """Report every PyTorch import under ``paths`` as ``<path>:<lineno>: <statement>``.

    ``paths`` holds files, directories (scanned recursively), or glob patterns; a missing path is ignored so that a
    pilot module that does not exist yet does not break the scan.  ``allowlist`` names the files allowed to import
    PyTorch anyway, and ``exclude`` prunes vendored trees.  The result is empty when the scanned surface is clean.
    """
    allowed = {_as_path(entry) for entry in allowlist}
    violations: list[str] = []
    for path in _iter_python_files(paths, exclude=exclude):
        if path in allowed:
            continue
        violations.extend(f"{_label(path)}:{lineno}: {statement}" for lineno, statement in _torch_imports(path))
    return violations


def assert_no_torch_imports(paths, *, allowlist=(), exclude=()) -> None:
    """:func:`torch_import_violations`, raising an :class:`AssertionError` naming every offending statement."""
    files = _iter_python_files(paths, exclude=exclude)
    if not files:
        raise AssertionError(f"None of the paths to scan for PyTorch imports exists: {sorted(map(str, paths))}")
    violations = torch_import_violations(paths, allowlist=allowlist, exclude=exclude)
    allowed = {_as_path(entry) for entry in allowlist}
    if violations:
        raise AssertionError(
            f"PyTorch is imported on the torch-free training path ({len(violations)} violation(s), "
            f"{len(files)} file(s) scanned, {len(allowed)} allowlisted):\n  " + "\n  ".join(violations)
        )


def has_gpu() -> bool:
    """Whether a GPU-backed Genesis run is plausible on this machine.

    The management library answers first (see `tests/gpu_info.py`); where neither NVML nor AMD SMI is installed, the
    kernel driver's device nodes still tell a GPU machine apart from a CPU-only one, exactly as the test session's
    worker sizing does.  Nothing here imports torch or quadrants: the parent process must stay out of the picture, and
    the skip decision must not need the very dependency the child under test is running without.
    """
    try:
        from tests.gpu_info import detect_gpu_backend
    except ImportError:  # Imported as a top-level module, e.g. from a plain interpreter with 'tests/' on sys.path
        from gpu_info import detect_gpu_backend

    if detect_gpu_backend() is not None:
        return True
    if not sys.platform.startswith("linux"):
        return False
    return any(Path(node).exists() for node in GPU_DEVICE_NODES)
