"""Detection and repair of the broken ROCm HSA runtime bundled with PyTorch.

On some AMD hosts PyTorch ships a ``libhsa-runtime64.so`` that silently corrupts AQL packets during long ROCm
rollouts, aborting the process with ``HSA_STATUS_ERROR_INVALID_PACKET_FORMAT``. The system ROCm runtime does not have
this defect, but it must be preloaded through ``LD_PRELOAD`` so that it is mapped before the dynamic loader resolves
PyTorch's HIP/HSA libraries. A ``ctypes`` preload at runtime is not equivalent because the libraries are already
mapped by then, hence the interpreter has to be re-executed once.

:func:`inspect_hsa_runtime` is pure detection and never re-executes, imports torch, or raises.
:func:`ensure_system_hsa_runtime` is the repair entry point, designed to be called once from ``gs.init`` before any
GPU work; it re-executes the current process when (and only when) doing so is safe and necessary.
Set ``GS_HSA_REEXEC=0`` to opt out; ``GS_HSA_REEXEC_DONE=1`` is the one-shot sentinel that prevents exec loops.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import logging
import os
import re
import sys
from pathlib import Path

from ..constants import backend as gs_backend

__all__ = ["HsaRuntimeStatus", "inspect_hsa_runtime", "system_hsa_runtime_path", "ensure_system_hsa_runtime"]

_HSA_LIB_NAME = "libhsa-runtime64.so"
_HIP_LIB_NAME = "libamdhip64.so"
_HSA_VERSIONED_GLOB = "libhsa-runtime64.so.1.*"
_HSA_VERSION_FILENAME_RE = re.compile(r"^libhsa-runtime64\.so\.([0-9][0-9.]*)$")
_HSA_VERSION_BYTES_RE = re.compile(rb"libhsa-runtime64\.so\.([0-9][0-9.]*)")
# The bundled runtime embeds its version string in the first few megabytes of the binary.
_VERSION_SCAN_BYTES = 16 * 1024 * 1024

_ENV_OPT_OUT = "GS_HSA_REEXEC"
_ENV_SENTINEL = "GS_HSA_REEXEC_DONE"
_ENV_PRELOAD = "LD_PRELOAD"

_LOG_PREFIX = "[genesis]"


@dataclasses.dataclass(frozen=True)
class HsaRuntimeStatus:
    """Outcome of inspecting the HSA runtimes bundled with torch vs. installed on the system.

    ``applicable`` is True only on a Linux ROCm PyTorch installation that bundles both the HSA runtime and the HIP
    library. ``needs_reexec`` states whether a re-exec is technically required (the system runtime differs and is not
    already preloaded); it is False once the one-shot sentinel is set. ``blocked_reason`` explains why an otherwise
    needed re-exec must not happen, and is None when nothing blocks it.
    """

    applicable: bool
    needs_reexec: bool
    reason: str
    bundled_path: Path | None = None
    bundled_version: str | None = None
    bundled_sha256: str | None = None
    system_path: Path | None = None
    system_version: str | None = None
    system_sha256: str | None = None
    preload: str | None = None
    preload_names_system: bool = False
    blocked_reason: str | None = None


def inspect_hsa_runtime() -> HsaRuntimeStatus:
    """Inspect the torch-bundled and system HSA runtimes without side effects.

    Torch is never imported: its location is taken from ``sys.modules`` when already loaded, or from
    ``importlib.util.find_spec`` otherwise. All failures are reported as a non-applicable status instead of raising.

    Returns:
        The detected status. ``needs_reexec`` is True only when a differing system runtime exists and the current
        process has not preloaded it yet and has not already been re-executed.
    """
    try:
        return _inspect_hsa_runtime()
    except Exception as e:
        return HsaRuntimeStatus(
            applicable=False,
            needs_reexec=False,
            reason=f"HSA runtime inspection failed unexpectedly: {e}",
        )


def system_hsa_runtime_path() -> Path | None:
    """Return the system HSA runtime that differs from the one bundled with torch, if any.

    Returns:
        The real path of the first differing ``libhsa-runtime64.so.1.*`` found in the ROCm installation directories,
        or None when the platform is not Linux, torch does not bundle a ROCm HSA runtime, or no differing candidate
        exists.
    """
    try:
        if sys.platform != "linux":
            return None
        bundled_path = _torch_lib_dir()
        if bundled_path is None:
            return None
        bundled_hsa_path = bundled_path / _HSA_LIB_NAME
        if not bundled_hsa_path.is_file() or not (bundled_path / _HIP_LIB_NAME).is_file():
            return None
        candidate = _find_system_candidate(bundled_hsa_path, _sha256(bundled_hsa_path))
        return None if candidate is None else candidate[0]
    except Exception:
        return None


def ensure_system_hsa_runtime(requested_backend=None, logger=None) -> bool:
    """Re-execute the current process with the system HSA runtime preloaded when that is required and safe.

    This must be called from a fresh interpreter, before any torch/quadrants GPU work, typically at the very start of
    ``gs.init``. It never raises and never returns True: a successful re-exec never returns, and every other path
    returns False.

    Args:
        requested_backend: The backend requested from ``gs.init``, a member of ``genesis.constants.backend``. The
            check is skipped entirely for ``cpu`` and ``metal``.
        logger: Optional logger exposing ``.info`` and ``.warning``. When omitted, the stdlib logger named after this
            module is used for warnings and the informational re-exec line is printed to stdout.

    Returns:
        Always False, since the process either continues without re-executing or is replaced by ``os.execve``.
    """
    try:
        return _ensure_system_hsa_runtime(requested_backend, logger)
    except Exception as e:
        _emit_warning(logger, f"Automatic HSA runtime re-exec failed unexpectedly: {e}. Continuing without it.")
        return False


def _ensure_system_hsa_runtime(requested_backend, logger) -> bool:
    if requested_backend is gs_backend.cpu or requested_backend is gs_backend.metal:
        return False

    if os.environ.get(_ENV_OPT_OUT) == "0":
        status = inspect_hsa_runtime()
        if status.applicable and status.system_path is not None:
            _emit_warning(
                logger,
                f"{_ENV_OPT_OUT}=0 disables the automatic HSA runtime re-exec: torch's bundled HSA runtime "
                f"{status.bundled_path} (version {_version_text(status.bundled_version)}) will be used instead of the "
                f"system one {status.system_path} (version {_version_text(status.system_version)}).",
            )
        return False

    if os.environ.get(_ENV_SENTINEL) == "1":
        return False

    status = inspect_hsa_runtime()
    if not status.applicable or status.system_path is None or status.preload_names_system:
        return False

    warn_only_reason = _warn_only_context()
    if warn_only_reason is not None:
        _emit_warning(
            logger,
            f"Skipping the automatic HSA runtime re-exec because the process is {warn_only_reason}: torch's bundled "
            f"HSA runtime {status.bundled_path} (version {_version_text(status.bundled_version)}) will be used "
            f"instead of the system one {status.system_path} (version {_version_text(status.system_version)}). "
            f"Launch a plain Python script to let it apply.",
        )
        return False

    if _torch_gpu_initialized():
        _emit_warning(
            logger,
            f"Skipping the automatic HSA runtime re-exec because torch GPU support is already initialized in this "
            f"process: torch's bundled HSA runtime {status.bundled_path} (version "
            f"{_version_text(status.bundled_version)}) will be used instead of the system one {status.system_path} "
            f"(version {_version_text(status.system_version)}). The re-exec must happen before any GPU work, so "
            f"restart the process to apply it.",
        )
        return False

    script_path = os.path.abspath(sys.argv[0])
    _emit_info(
        logger,
        f"Re-executing the interpreter with the system HSA runtime {status.system_path} (version "
        f"{_version_text(status.system_version)}) preloaded instead of torch's bundled {status.bundled_path} (version "
        f"{_version_text(status.bundled_version)}), because torch's bundled HSA runtime corrupts AQL packets on long "
        f"ROCm rollouts. Set {_ENV_OPT_OUT}=0 to opt out.",
    )

    env = dict(os.environ)
    # The sentinel must be set in the parent environment so that the child inherits it and never loops.
    env[_ENV_SENTINEL] = "1"
    existing_preload = env.get(_ENV_PRELOAD, "")
    env[_ENV_PRELOAD] = f"{status.system_path}:{existing_preload}" if existing_preload else str(status.system_path)

    _flush_all(logger)
    # argv[0] must stay the interpreter path: CPython treats the first argument after argv[0] as the script to run.
    exec_argv = [sys.executable, script_path, *sys.argv[1:]]
    try:
        os.execve(sys.executable, exec_argv, env)
    except OSError as e:
        _emit_warning(
            logger,
            f"Failed to re-execute the interpreter with the system HSA runtime {status.system_path}: {e}. "
            f"Falling back to torch's bundled HSA runtime {status.bundled_path}.",
        )
    return False


def _inspect_hsa_runtime() -> HsaRuntimeStatus:
    if sys.platform != "linux":
        return HsaRuntimeStatus(
            applicable=False,
            needs_reexec=False,
            reason="The HSA runtime preload repair is only supported on Linux.",
        )

    torch_lib_dir = _torch_lib_dir()
    if torch_lib_dir is None:
        return HsaRuntimeStatus(
            applicable=False,
            needs_reexec=False,
            reason="The PyTorch installation could not be located, so its bundled HSA runtime is unknown.",
        )

    bundled_path = torch_lib_dir / _HSA_LIB_NAME
    if not bundled_path.is_file() or not (torch_lib_dir / _HIP_LIB_NAME).is_file():
        return HsaRuntimeStatus(
            applicable=False,
            needs_reexec=False,
            reason="The installed PyTorch does not bundle an AMD ROCm HSA runtime, so no preload repair is needed.",
        )

    bundled_sha256 = _sha256(bundled_path)
    bundled_version = _version_from_filename(bundled_path)
    if bundled_version is None:
        bundled_version = _scan_embedded_version(bundled_path)

    candidate = _find_system_candidate(bundled_path, bundled_sha256)
    system_path = None if candidate is None else candidate[0]
    system_sha256 = None if candidate is None else candidate[1]
    system_version = None if system_path is None else _version_from_filename(system_path)

    preload = os.environ.get(_ENV_PRELOAD)
    preload_names_system = (
        system_path is not None and _preload_names_path(preload, system_path, system_sha256)
    )

    sentinel_set = os.environ.get(_ENV_SENTINEL) == "1"
    opt_out = os.environ.get(_ENV_OPT_OUT) == "0"
    needs_reexec = system_path is not None and not preload_names_system and not sentinel_set

    if system_path is None:
        reason = "No system HSA runtime differing from the one bundled with PyTorch was found, so no re-exec is needed."
    elif preload_names_system:
        reason = (
            f"The system HSA runtime {system_path} is already preloaded, so no re-exec is required."
        )
    elif sentinel_set:
        reason = (
            f"A re-exec already happened ({_ENV_SENTINEL}=1), so no further re-exec is required to preload the "
            f"system HSA runtime {system_path}."
        )
    else:
        reason = (
            f"PyTorch bundles HSA runtime {bundled_path} (version {_version_text(bundled_version)}, sha256 "
            f"{bundled_sha256}) that differs from the system one {system_path} (version "
            f"{_version_text(system_version)}, sha256 {system_sha256}), so a re-exec preloading the system runtime is "
            f"required."
        )

    blocked_reason = None
    if system_path is not None and not preload_names_system:
        if sentinel_set:
            blocked_reason = f"A re-exec already happened ({_ENV_SENTINEL}=1)."
        elif opt_out:
            blocked_reason = f"The re-exec is disabled by {_ENV_OPT_OUT}=0."

    return HsaRuntimeStatus(
        applicable=True,
        needs_reexec=needs_reexec,
        reason=reason,
        bundled_path=bundled_path,
        bundled_version=bundled_version,
        bundled_sha256=bundled_sha256,
        system_path=system_path,
        system_version=system_version,
        system_sha256=system_sha256,
        preload=preload,
        preload_names_system=preload_names_system,
        blocked_reason=blocked_reason,
    )


def _torch_lib_dir() -> Path | None:
    """Locate torch's ``lib`` directory without importing torch."""
    torch_module = sys.modules.get("torch")
    if torch_module is not None:
        torch_file = torch_module.__file__
        if torch_file is None:
            return None
        return Path(torch_file).parent / "lib"
    try:
        spec = importlib.util.find_spec("torch")
    except (ImportError, ValueError):
        return None
    if spec is None or spec.origin is None:
        return None
    return Path(spec.origin).parent / "lib"


def _candidate_lib_dirs() -> list[Path]:
    """List the ROCm library directories to search, in priority order and without duplicates."""
    roots = []
    for env_var in ("ROCM_PATH", "ROCM_HOME"):
        root = os.environ.get(env_var)
        if root:
            roots.append(Path(root))
    roots.append(Path("/opt/rocm"))
    roots.extend(sorted(Path("/opt").glob("rocm-*")))

    lib_dirs = []
    seen_realpaths = set()
    for root in roots:
        for subdir in ("lib", "lib64"):
            lib_dir = root / subdir
            real_path = os.path.realpath(lib_dir)
            if real_path in seen_realpaths:
                continue
            seen_realpaths.add(real_path)
            lib_dirs.append(lib_dir)
    return lib_dirs


def _find_system_candidate(bundled_path: Path, bundled_sha256: str | None) -> tuple[Path, str] | None:
    """Find the first system HSA runtime whose real path and sha256 differ from torch's bundled one."""
    bundled_real_path = os.path.realpath(bundled_path)
    seen_realpaths = {bundled_real_path}
    for lib_dir in _candidate_lib_dirs():
        if not lib_dir.is_dir():
            continue
        candidates = list(sorted(lib_dir.glob(_HSA_VERSIONED_GLOB)))
        candidates.extend((lib_dir / f"{_HSA_LIB_NAME}.1", lib_dir / _HSA_LIB_NAME))
        for candidate in candidates:
            if not candidate.is_file():
                continue
            logical_path = _logical_path(candidate)
            real_path = os.path.realpath(logical_path)
            if real_path in seen_realpaths:
                continue
            seen_realpaths.add(real_path)
            sha256 = _sha256(logical_path)
            if sha256 is None or sha256 == bundled_sha256:
                continue
            return logical_path, sha256
    return None


def _logical_path(path: Path) -> Path:
    """Resolve the file's own symlinks while keeping the (possibly symlinked) directory it was found in."""
    real_path = Path(os.path.realpath(path))
    logical_path = path.parent / real_path.name
    return logical_path if logical_path.is_file() else real_path


def _preload_names_path(preload: str | None, system_path: Path, system_sha256: str | None) -> bool:
    """Check whether ``LD_PRELOAD`` already contains the given system library."""
    if not preload:
        return False
    system_real_path = os.path.realpath(system_path)
    for entry in re.split(r"[:\s]+", preload):
        if not entry:
            continue
        if os.path.realpath(entry) == system_real_path:
            return True
        if system_sha256 is not None and _sha256(Path(entry)) == system_sha256:
            return True
    return False


def _version_from_filename(path: Path) -> str | None:
    match = _HSA_VERSION_FILENAME_RE.match(path.name)
    return None if match is None else match.group(1)


def _scan_embedded_version(path: Path) -> str | None:
    """Read the version string embedded in the binary, without shelling out to strings/readelf."""
    try:
        with path.open("rb") as f:
            head = f.read(_VERSION_SCAN_BYTES)
    except OSError:
        return None
    match = _HSA_VERSION_BYTES_RE.search(head)
    return None if match is None else match.group(1).decode("ascii")


def _sha256(path: Path) -> str | None:
    try:
        with path.open("rb") as f:
            return hashlib.file_digest(f, "sha256").hexdigest()
    except OSError:
        return None


def _warn_only_context() -> str | None:
    """Return why a re-exec would be unsafe in this process, or None when it is safe."""
    if "PYTEST_VERSION" in os.environ or "pytest" in sys.modules:
        return "running under pytest"
    if sys.flags.interactive:
        return "running in an interactive interpreter"
    main_module = sys.modules.get("__main__")
    if main_module is not None and main_module.__spec__ is not None:
        return "launched via 'python -m'"
    if not sys.argv or not sys.argv[0] or not os.path.isfile(sys.argv[0]):
        return "running without a real script path in sys.argv[0]"
    if "IPython" in sys.modules or "ipykernel" in sys.modules:
        return "running inside IPython/ipykernel"
    return None


def _torch_gpu_initialized() -> bool:
    """Check whether torch has already initialized its GPU context, without importing torch."""
    torch_module = sys.modules.get("torch")
    if torch_module is None:
        return False
    try:
        return bool(torch_module.cuda.is_initialized())
    except Exception:
        return False


def _version_text(version: str | None) -> str:
    return "unknown" if version is None else version


def _emit_info(logger, message: str) -> None:
    line = f"{_LOG_PREFIX} {message}"
    if logger is not None:
        logger.info(line)
        return
    print(line, flush=True)


def _emit_warning(logger, message: str) -> None:
    line = f"{_LOG_PREFIX} {message}"
    if logger is not None:
        logger.warning(line)
        return
    logging.getLogger(__name__).warning(line)


def _flush_all(logger) -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except (OSError, ValueError):
            pass
    handlers = list(logging.root.handlers)
    if isinstance(logger, logging.Logger):
        handlers.extend(logger.handlers)
    for handler in handlers:
        try:
            handler.flush()
        except (OSError, ValueError):
            pass
