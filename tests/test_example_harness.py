"""Cheap guards of the example smoke harness itself (``tests/test_examples.py``).

The harness collects every ``examples/**/*.py`` script that is not a shared helper module and runs it as a subprocess,
but it is marked ``examples``, so the default ``-m required`` selection never exercises it: a broken ignore pattern or a
missing ``EXAMPLE_DEPENDENCIES`` gate would only surface in the opt-in ``examples`` CI job, where every failure costs
minutes. This module pins the harness' invariants with pure file/AST work instead: no scene, no GPU, no subprocess,
and the examples themselves are never imported.
The three properties are the ones that have already gone wrong or nearly so:
- the device graphs/rollouts (``*_device_kernels.py``, ``*_device_rollout.py``) have no ``__main__`` demo and must never
  be executed, in particular the drone ones, which are written independently of this harness;
- discovery must still return the real demos of those directories, so an over-broad pattern cannot silently empty the
  collected set;
- an example that only reaches ``evoxc``/``tensorneat``/``tensordict`` through a local sibling module must still be
  gated, which is why the scan follows local imports transitively instead of only the script's own imports.
"""
import ast
from pathlib import Path

import pytest

from tests.test_examples import (
    EXAMPLES_DIR,
    EXAMPLE_DEPENDENCIES,
    IGNORE_SCRIPT_PATTERNS,
    _discover_examples,
    _is_ignored_script,
)


pytestmark = [pytest.mark.required, pytest.mark.parametrize("backend", [None])]  # Disable genesis initialization

#: Shared device/kernel helpers of the Go2 and drone stacks, relative to 'examples/'. None of them is a runnable demo.
DEVICE_HELPER_MODULES = [
    "locomotion/go2_device_kernels.py",
    "locomotion/go2_device_rollout.py",
    "locomotion/go2_neat_kernels.py",
    "locomotion/go2_evox_kernels.py",
    "locomotion/go2_legacy_reference.py",
    "drone/hover_device_rollout.py",
    "drone/hover_kernels.py",
    "drone/hover_legacy_reference.py",
]
#: Written independently of this harness, so only the ignore pattern is asserted, never that the file exists.
PENDING_DEVICE_HELPER_MODULES = [
    "drone/hover_device_kernels.py",
]
#: Demos of the two device-helper directories, which the generic helper patterns must not swallow.
ANCHOR_DEMOS = [
    "locomotion/try1.py",
    "locomotion/go2_backflip.py",
    "drone/hover_neat_check.py",
]
#: Optional third-party modules that an example may reach through a local sibling instead of importing them itself.
GATED_MODULES = ("evoxc", "tensorneat", "tensordict")


def _imported_modules(path: Path) -> set[str]:
    """Every absolute module name ``path`` imports, at any nesting depth (function-level imports included)."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            # A relative import can only reach a sibling of the importing module, never an installed package
            modules.add(node.module)
    return modules


def _local_module_index() -> dict[str, list[Path]]:
    """Map every module name the examples can import by bare name to the example files providing it."""
    index: dict[str, list[Path]] = {}
    for path in sorted(EXAMPLES_DIR.rglob("*.py")):
        index.setdefault(path.stem, []).append(path)
    return index


def _reached_modules(path: Path, index: dict[str, list[Path]], seen: set[Path] | None = None) -> set[str]:
    """The subset of ``GATED_MODULES`` reachable from ``path`` through its own and its local siblings' imports."""
    seen = set() if seen is None else seen
    if path in seen:
        return set()
    seen.add(path)

    reached = set()
    for module in _imported_modules(path):
        top_level = module.split(".")[0]
        if top_level in GATED_MODULES:
            reached.add(top_level)
        else:
            # Each script's own directory lands on 'sys.path', so local helpers are imported by bare module name
            for local_path in index.get(top_level, ()):
                reached.update(_reached_modules(local_path, index, seen))
    return reached


def _dependency_gate(path: Path) -> list[str]:
    """The optional dependencies the harness requires for ``path``: its own name, then every parent directory key."""
    rel = path.relative_to(EXAMPLES_DIR)
    gate = list(EXAMPLE_DEPENDENCIES.get(rel.name, []))
    for parent in rel.parents:
        if parent != Path("."):
            gate.extend(EXAMPLE_DEPENDENCIES.get(parent.as_posix(), []))
    return gate


def test_discovered_examples_are_runnable_python_files():
    examples = _discover_examples()
    discovered = {path.relative_to(EXAMPLES_DIR).as_posix() for path in examples}

    assert examples, f"discovery returned no example from '{EXAMPLES_DIR}'"
    for path in examples:
        assert path.is_file(), f"discovered example '{path}' does not exist"
        assert path.suffix == ".py", f"discovered example '{path}' is not a Python file"
        assert not _is_ignored_script(path), f"discovery returned the ignored script '{path}'"
    for rel_path in ANCHOR_DEMOS:
        assert rel_path in discovered, f"'{rel_path}' is a runnable demo but was not discovered"


def test_device_helper_modules_are_never_discovered():
    discovered = {path.relative_to(EXAMPLES_DIR).as_posix() for path in _discover_examples()}

    for rel_path in DEVICE_HELPER_MODULES + PENDING_DEVICE_HELPER_MODULES:
        # Assert the pattern rather than the file: the pending drone helper may not be committed yet
        assert _is_ignored_script(EXAMPLES_DIR / rel_path), f"'{rel_path}' is not matched by IGNORE_SCRIPT_PATTERNS"
        assert rel_path not in discovered, f"the device helper module '{rel_path}' would be run as a demo"


def test_discovered_examples_gate_their_optional_imports():
    index = _local_module_index()
    gated = {}

    for path in _discover_examples():
        rel_path = path.relative_to(EXAMPLES_DIR).as_posix()
        reached = _reached_modules(path, index)
        missing = sorted(reached - set(_dependency_gate(path)))
        if missing:
            pytest.fail(
                f"example '{rel_path}' imports {missing} (directly or through a local helper) but "
                f"EXAMPLE_DEPENDENCIES is missing that gate on '{path.name}' or '{path.parent.name}'"
            )
        if reached:
            gated[rel_path] = _dependency_gate(path)

    # A scan that reaches nothing would pass vacuously, e.g. after the gate lookup or the AST walk is broken
    assert gated, f"no discovered example reaches any of {GATED_MODULES}, so the gate check is vacuous"
    for rel_path, gate in gated.items():
        assert gate, f"example '{rel_path}' reaches a gated module, but its dependency gate is empty"
    # The two device stacks are the reason this check exists: both must be found through their local helpers
    assert "locomotion/try1.py" in gated, "the 'evoxc' import of 'go2_evox_kernels.py' was not traced from 'try1.py'"
