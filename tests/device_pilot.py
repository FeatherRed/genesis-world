"""Task-agnostic helpers for pilot-level device-residency tests.

The pilots under ``examples/`` (Go2 walking, the drone, ...) each carry a host (numpy) reference and a
device-resident (``evoxc``) implementation of one loop, so their parity tests all need the same four
things: comparing a device value with a host one, probing a reachable accelerator, and skipping
politely when the pilot module or the hardware is not there yet.  This module is the single place
those live, so a new pilot's test module imports them instead of re-implementing them.

Nothing here imports ``evoxc``, ``genesis`` or ``torch`` at module scope: the device probe imports
``evoxc`` inside the function, so this module stays importable on a host where none of them exist and
its import has no side effect.

Conventions
-----------
* :func:`as_host` accepts every value a pilot hands across a boundary -- an ``evoxc`` tensor (moved
  with its payload's own ``to_host``, or through ``.to(cpu:0)``), a ``genesis.utils.device_buffer.
  DeviceView`` (``to_numpy``), a ``torch`` tensor (when the parent process has one),
  a DLPack producer and a plain numpy array/scalar -- and always returns a numpy array.
* :func:`max_abs_diff` / :func:`max_rel_diff` ignore non-finite entries, which is how the pilots'
  NaN-padded genome/genotype slots stay comparable (``verify_device.py``'s convention).
* :func:`assert_allclose` reports the *measured* maximum absolute and relative difference in its
  failure message, so a regression says how far it moved instead of only that it moved.
* :func:`reachable_device_or_skip` never trusts an enumeration: ``evoxc.devices()`` lists every
  runtime the wheel was built with, so a ROCm-only host also enumerates ``cuda:0``.  The only honest
  reachability test is moving one float32 element there through the backend's placement provider.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

__all__ = [
    "DEVICE_ENV_VAR",
    "as_host",
    "assert_allclose",
    "max_abs_diff",
    "max_rel_diff",
    "reachable_device_or_skip",
    "skip_unless_torch_free",
    "torch_import_offenders",
    "torch_imports",
]

#: Environment variable an explicit device request comes from (``"rocm:0"``, ``"cuda:1"``, ...).
DEVICE_ENV_VAR = "EVOXC_DEVICE"

#: The device ``(backend, device)`` pairs probed in this process, keyed by probe request.
_DEVICE_CACHE: dict[tuple[str, str], tuple[str, object]] = {}


def as_host(value):
    """Materialize one device or host value as a numpy array.

    The value is never copied more than the move requires: an ``evoxc`` tensor hands over its
    payload's own readback (``tensor.data.to_host()``), which is what ``go2_device_rollout`` uses;
    an ``evoxc`` tensor on a non-CPU device is moved to ``cpu:0`` first; a ``DeviceView`` uses
    ``to_numpy()``; a ``torch`` tensor (only reachable in a process that has PyTorch, e.g. the
    parity tests that also drive the numpy reference) is detached and moved to the host; numpy
    arrays, numpy scalars and Python scalars pass straight through ``np.asarray``.

    Args:
        value: An ``evoxc`` tensor, a ``DeviceView``, a ``torch`` tensor, a DLPack producer, or a
            host array/scalar.

    Returns:
        np.ndarray: The host value; a 0-d array for a scalar tensor.
    """
    if isinstance(value, np.ndarray):
        return value
    data = getattr(value, "data", None)
    to_host = getattr(data, "to_host", None)
    if callable(to_host):
        return np.asarray(to_host())
    to_numpy = getattr(value, "to_numpy", None)
    if callable(to_numpy):
        return np.asarray(to_numpy())
    detach = getattr(value, "detach", None)
    if callable(detach):
        return np.asarray(detach().cpu().numpy())
    if hasattr(value, "device") and hasattr(value, "numpy"):
        if getattr(value.device, "kind", "cpu") != "cpu":
            import evoxc

            value = value.to(evoxc.Device("cpu", 0))
        return np.asarray(value.numpy())
    dlpack = getattr(value, "__dlpack__", None)
    if callable(dlpack):
        return np.from_dlpack(value)
    return np.asarray(value)


def max_abs_diff(actual, reference) -> float:
    """Largest finite absolute difference between two values; ``inf`` on a shape mismatch.

    Non-finite entries are ignored, so the NaN padding of an unused genome slot never decides the
    answer (the convention of ``verify_device.float_max_abs_diff``).
    """
    left = np.asarray(as_host(actual), dtype=np.float64)
    right = np.asarray(as_host(reference), dtype=np.float64)
    if left.shape != right.shape:
        return float("inf")
    with np.errstate(invalid="ignore"):
        differences = np.abs(left - right)
    differences = differences[np.isfinite(differences)]
    return float(differences.max()) if differences.size else 0.0


def max_rel_diff(actual, reference) -> float:
    """Largest relative difference against the magnitude of ``reference``; ``inf`` on a mismatch.

    Where the reference is zero the absolute difference is reported instead (the convention of
    ``verify_device.float_max_rel_diff``).
    """
    left = np.asarray(as_host(actual), dtype=np.float64)
    right = np.asarray(as_host(reference), dtype=np.float64)
    if left.shape != right.shape:
        return float("inf")
    with np.errstate(invalid="ignore", divide="ignore"):
        differences = np.abs(left - right)
        scale = np.abs(right)
        relative = np.where(scale > 0.0, differences / np.where(scale > 0.0, scale, 1.0), differences)
    relative = relative[np.isfinite(relative)]
    return float(relative.max()) if relative.size else 0.0


def assert_allclose(actual, reference, *, atol, rtol, label) -> tuple[float, float]:
    """Assert ``actual`` is within ``atol + rtol * |reference|``; return the measured differences.

    The measured maximum absolute and relative differences are part of the failure message, so a
    regression reports how far the value moved instead of only that it moved.

    Args:
        actual: The value under test (device or host).
        reference: The reference value (device or host).
        atol (float): Absolute tolerance.
        rtol (float): Relative tolerance, scaled by ``|reference|``.
        label (str): What is being compared, used in the failure message.

    Returns:
        tuple: ``(max_abs_diff, max_rel_diff)`` measured between the two values.
    """
    left = np.asarray(as_host(actual), dtype=np.float64)
    right = np.asarray(as_host(reference), dtype=np.float64)
    abs_diff = max_abs_diff(left, right)
    rel_diff = max_rel_diff(left, right)
    assert np.allclose(left, right, atol=atol, rtol=rtol, equal_nan=True), (
        f"{label}: max |diff| {abs_diff:.3e}, max rel {rel_diff:.3e}, outside atol={atol:.1e} + "
        f"rtol={rtol:.1e} x |reference| (largest |reference| "
        f"{float(np.max(np.abs(right))) if right.size else 0.0:.4g}; shapes {left.shape} vs {right.shape})"
    )
    return abs_diff, rel_diff


def reachable_device_or_skip(backend="iree"):
    """Return ``(backend, device)`` of a reachable non-CPU device, or skip the calling test.

    An explicit :data:`DEVICE_ENV_VAR` request wins; it is still probed, so a device a test session
    was pointed at but that is unusable skips with the failure instead of crashing mid-rollout.  With
    no request, every non-CPU device ``evoxc.devices()`` enumerates is probed in order by moving one
    float32 element there through the backend's placement provider.

    The probe is cached per process: the first enumeration of the IREE devices costs several seconds.

    Args:
        backend (str): The ``evoxc`` backend the caller wants the device for.

    Returns:
        tuple: ``(backend, evoxc.Device)`` of a device that accepted a write.
    """
    requested = __import__("os").environ.get(DEVICE_ENV_VAR, "").strip()
    cache_key = (str(backend), requested)
    cached = _DEVICE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    import evoxc

    evoxc.backends.get(backend)
    if requested:
        kind, _, index = requested.partition(":")
        candidates = [evoxc.Device(kind.strip(), int(index) if index.strip() else 0)]
    else:
        candidates = [device for device in evoxc.devices() if device.kind != "cpu"]

    unusable = []
    for device in candidates:
        try:
            evoxc.from_numpy(np.zeros(1, dtype=np.float32)).to(device)
        except Exception as exc:  # noqa: BLE001 - an unreachable runtime is an expected outcome to skip on
            unusable.append(f"{device.kind}:{device.index} ({type(exc).__name__})")
            continue
        _DEVICE_CACHE[cache_key] = (str(backend), device)
        return _DEVICE_CACHE[cache_key]
    pytest.skip(
        f"no reachable non-CPU evoxc device for backend {backend!r}"
        + (f" (requested through {DEVICE_ENV_VAR}={requested!r})" if requested else "")
        + f"; probed {', '.join(unusable) or 'no non-CPU device'}"
    )


def torch_imports(path) -> list[str]:
    """Return the ``torch`` modules ``path`` imports, from an AST scan; empty when unparsable.

    The file is parsed, never imported: importing a pilot module needs ``torch``, ``genesis`` and a
    GPU.  An unreadable or unparsable file yields an empty list, so the detector cannot fail the
    suite itself; a *missing* file is reported by :func:`torch_import_offenders` instead.

    Args:
        path (str | pathlib.Path): The module to scan.

    Returns:
        list[str]: ``["torch", "torch.nn", ...]``, in source order (duplicates kept).
    """
    try:
        tree = ast.parse(Path(path).read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError):
        return []
    modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names if _is_torch_name(alias.name))
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and _is_torch_name(node.module):
            modules.append(node.module)
    return modules


def _is_torch_name(name) -> bool:
    """Whether a dotted module name is ``torch`` or a submodule of it."""
    return name is not None and (name == "torch" or name.startswith("torch."))


def torch_import_offenders(*paths) -> list[str]:
    """Describe why the given pilot modules are not Torch-free; empty when every one of them is.

    A missing module is an offender too: the entry point of a pilot that does not exist proves
    nothing, so a test that wants a Torch-free pilot must skip until it lands.

    Args:
        *paths: The pilot modules that a Torch-free claim covers.

    Returns:
        list[str]: One human-readable problem per offending module.
    """
    problems = []
    for entry in paths:
        path = Path(entry)
        if not path.exists():
            problems.append(f"{path.name} is missing")
            continue
        imports = torch_imports(path)
        if imports:
            problems.append(f"{path.name} imports {', '.join(sorted(set(imports)))}")
    return problems


def skip_unless_torch_free(*paths, feature="the pilot"):
    """Skip the calling test unless every given module exists and imports no PyTorch.

    Args:
        *paths: The pilot modules the Torch-free claim covers.
        feature (str): What the modules are, used in the skip reason.
    """
    problems = torch_import_offenders(*paths)
    if problems:
        pytest.skip(f"{feature} is not Torch-free yet: " + "; ".join(problems))
