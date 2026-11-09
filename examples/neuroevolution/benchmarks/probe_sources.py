"""Source-level facts of the Phase-4 probe: the AST boundary scan and the committed records it cites.

Two read-only concerns live here so :mod:`bench_probe` stays a CLI:

* the AST scan of the neuroevolution layer and the pilot modules -- which file imports ``torch`` /
  ``genesis`` / ``tensordict``, at module level or inside a function, with the line and scope.
  Files are parsed, never imported: importing ``go2_env``/``hover_env`` is exactly what the
  ``unmatched`` probe records as impossible;
* the structured ``interface_mismatch`` description of the two torch baselines, assembled from the
  same parse (constructor arguments, ``step`` returns, the internal reset, ``get_cfgs``);
* the extraction of the already-committed boundary numbers (per-step ``state_read`` share, per-step
  timer means, residency host calls/bytes/syncs) with the source filename and the record's git head,
  so nothing here is re-measured or invented.

Nothing in this module imports ``genesis``, ``torch``, ``tensordict`` or a pilot module.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import common
from bench_envs import _r

#: Directory of the committed records this module cites.
RESULTS_DIR = Path(__file__).resolve().parent / "results"

#: Import roots the torch/CPU boundary cares about.
TRACKED_ROOTS = ("torch", "genesis", "tensordict", "rsl_rl", "rsl_rl_lib", "quadrants", "evoxc", "tensorneat")

#: The two pre-migration baselines and where their modules live.
BASELINES = {
    "go2": {
        "env": "examples/locomotion/go2_env.py",
        "train": "examples/locomotion/go2_train.py",
        "env_class": "Go2Env",
        "env_module": "go2_env",
        "train_module": "go2_train",
        "import_dir": "examples/locomotion",
    },
    "hover": {
        "env": "examples/drone/hover_env.py",
        "train": "examples/drone/hover_train.py",
        "env_class": "HoverEnv",
        "env_module": "hover_env",
        "train_module": "hover_train",
        "import_dir": "examples/drone",
    },
}

#: The pilot modules the AST scan covers, in addition to every ``examples/neuroevolution/*.py``.
PILOT_SCAN_PATHS = (
    "examples/locomotion/go2_neat_env.py",
    "examples/locomotion/go2_neat_kernels.py",
    "examples/locomotion/go2_evox_kernels.py",
    "examples/locomotion/go2_device_rollout.py",
    "examples/drone/hover_neat_env.py",
    "examples/drone/hover_kernels.py",
    "examples/drone/hover_device_rollout.py",
)

_SOURCE_CACHE: dict[str, dict] = {}


# --- AST scan ---------------------------------------------------------------------
class _SourceScanner(ast.NodeVisitor):
    """Collect the tracked imports and the function signatures of one file, with their scope."""

    def __init__(self):
        self.imports: list[dict] = []
        self.functions: dict[str, dict] = {}
        self.classes: dict[str, dict] = {}
        self._scopes: list[tuple[str, str]] = []

    def _scope_name(self) -> str:
        return ":".join(name for _kind, name in self._scopes) or "module"

    def _qualified(self, name: str) -> str:
        prefix = ".".join(name for _kind, name in self._scopes)
        return f"{prefix}.{name}" if prefix else name

    def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802 - ast visitor naming
        self.classes[node.name] = {"line": node.lineno, "bases": [ast.unparse(base) for base in node.bases]}
        self._scopes.append(("class", node.name))
        self.generic_visit(node)
        self._scopes.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802 - ast visitor naming
        kind = "method" if self._scopes and self._scopes[-1][0] == "class" else "function"
        self.functions[self._qualified(node.name)] = {
            "line": node.lineno,
            "kind": kind,
            "args": [arg.arg for arg in node.args.args if arg.arg != "self"],
            "defaults": [ast.unparse(default) for default in node.args.defaults],
            "returns": [
                ast.unparse(statement.value)[:300]
                for statement in ast.walk(node)
                if isinstance(statement, ast.Return) and statement.value is not None
            ][:8],
        }
        self._scopes.append((kind, node.name))
        self.generic_visit(node)
        self._scopes.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Import(self, node: ast.Import) -> None:  # noqa: N802 - ast visitor naming
        for alias in node.names:
            self._record(node, alias.name, alias=alias.asname)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # noqa: N802 - ast visitor naming
        module = ("." * (node.level or 0)) + (node.module or "")
        self._record(node, module, names=[alias.name for alias in node.names])

    def _record(self, node, module: str, alias=None, names=None) -> None:
        root = module.split(".")[0].lstrip(".")
        if root not in TRACKED_ROOTS:
            return
        self.imports.append(
            {
                "module": module,
                "root": root,
                "line": node.lineno,
                "scope": self._scope_name(),
                "scope_kind": "module" if not self._scopes else self._scopes[-1][0],
                "statement": ast.unparse(node)[:300],
                "alias": alias,
                "names": names,
            }
        )


def _scan_source(relative_path: str) -> dict:
    """Parse one source file (cached) and return its tracked imports plus its function table."""
    if relative_path in _SOURCE_CACHE:
        return _SOURCE_CACHE[relative_path]
    path = common.repo_root() / relative_path
    entry = {"path": relative_path, "exists": path.is_file(), "imports": [], "functions": {}, "classes": {}}
    if entry["exists"]:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError) as exc:
            entry["syntax_error"] = f"{type(exc).__name__}: {exc}"
        else:
            scanner = _SourceScanner()
            scanner.visit(tree)
            entry.update(imports=scanner.imports, functions=scanner.functions, classes=scanner.classes)
    _SOURCE_CACHE[relative_path] = entry
    return entry


def _import_scopes(entry: dict, root: str) -> list[dict]:
    """Return the tracked imports of one root (``torch``/``genesis``/...) with their scope."""
    return [
        {"line": item["line"], "scope": item["scope"], "scope_kind": item["scope_kind"], "statement": item["statement"]}
        for item in entry["imports"]
        if item["root"] == root
    ]


def _call_facts(entry: dict, qualified: str | None) -> dict:
    """Return one function's recorded signature facts, or an ``absent`` marker."""
    if not qualified or qualified not in entry["functions"]:
        return {"present": False, "qualified": qualified}
    return {"present": True, "qualified": qualified, **entry["functions"][qualified]}


def interface_mismatch(task: str, observed_import: dict, observed_adapter: dict) -> dict:
    """Describe why the baseline env does not implement the batched-episode contract, from source."""
    spec = BASELINES[task]
    env = _scan_source(spec["env"])
    train = _scan_source(spec["train"])
    class_name = spec["env_class"]
    init = _call_facts(env, f"{class_name}.__init__")
    step = _call_facts(env, f"{class_name}.step")
    reset = _call_facts(env, f"{class_name}._reset_idx")
    if not reset["present"]:
        reset = _call_facts(env, f"{class_name}.reset_idx")
    get_cfgs = _call_facts(train, "get_cfgs")
    has_evaluate = any(name == "evaluate" or name.endswith(".evaluate") for name in env["functions"])
    module_imports = [
        {"root": item["root"], "module": item["module"], "line": item["line"], "statement": item["statement"]}
        for item in env["imports"]
        if item["scope_kind"] == "module"
    ]
    train_module_imports = [
        {"root": item["root"], "module": item["module"], "line": item["line"], "statement": item["statement"]}
        for item in train["imports"]
        if item["scope_kind"] == "module"
    ]
    exc = observed_import["exception"] or {}
    reasons = [
        f"import fails before construction: {exc.get('type')}: {exc.get('message')}",
        "the constructor needs env_cfg/obs_cfg/reward_cfg/command_cfg, produced by "
        f"{spec['train_module']}.get_cfgs() which is itself unimportable (it enforces rsl-rl-lib>=5.0.0)",
        "step() returns (TensorDict observation, rew_buf, reset_buf, extras); there is no "
        "episode-return array and returns only accumulate in the env's own episode_sums/extras",
        "auto-reset is internal (no per-episode seed argument), so policies cannot be scored on the "
        "same ordered episode seeds, and there is no episodes_per_policy row contract",
        "there is no evaluate(pop_arrays, generation) entry point, so the trainer cannot drive it",
    ]
    return {
        "task": task,
        "attempted": True,
        "blocked_at": "import",
        "observed_import_error": observed_import["exception"],
        "observed_adapter_error": observed_adapter["exception"],
        "sources": {"env": spec["env"], "train": spec["train"]},
        "env_module_level_imports": module_imports,
        "train_module_level_imports": train_module_imports,
        "constructor": init,
        "step": step,
        "auto_reset": reset,
        "config_source": get_cfgs,
        "has_evaluate": has_evaluate,
        "contract": {
            "rows": "none: the env steps one homogeneous batch of num_envs rows",
            "evaluate": "absent" if not has_evaluate else "unexpectedly present (inspect the source)",
            "episode_return_boundary": "absent: returns live in the env's episode_sums/extras, never as a per-row episode return",
            "per_episode_seed": "absent: the internal reset takes only envs_idx, with no seed",
            "step_return_value": "step(actions) -> (TensorDict observation, rew_buf, reset_buf, extras)",
        },
        "required_for_adapter": [
            "tensordict (module-level TensorDict observation)",
            "rsl-rl-lib>=5.0.0 (the config source go2_train/hover_train enforces the version and imports rsl_rl)",
            "a per-environment seedable reset wrapper (the internal reset takes no seed)",
            "an episode-return boundary and an evaluate(pop_arrays, generation) implementation",
        ],
        "reasons": reasons,
        "verdict": (
            "unmatched baseline, not a measured one: the import fails, and even with the dependencies "
            "installed the env lacks the row/seed/episode-return/evaluate contract the trainer needs"
        ),
    }


# --- torch / CPU boundary scan ----------------------------------------------------
def scan_targets() -> list[str]:
    """Return the neuroevolution modules plus the named pilot modules."""
    root = common.repo_root()
    targets = sorted(str(path.relative_to(root)) for path in (root / "examples/neuroevolution").glob("*.py"))
    targets += [path for path in PILOT_SCAN_PATHS if path not in targets]
    return targets


def torch_scan() -> dict:
    """AST-scan every boundary module and summarise where ``torch``/``genesis``/``tensordict`` enter."""
    files = {}
    for relative in scan_targets():
        entry = _scan_source(relative)
        files[relative] = {
            "exists": entry["exists"],
            "syntax_error": entry.get("syntax_error"),
            "torch": _import_scopes(entry, "torch"),
            "genesis": _import_scopes(entry, "genesis"),
            "tensordict": _import_scopes(entry, "tensordict"),
            "rsl_rl": _import_scopes(entry, "rsl_rl"),
        }
    at_import = [path for path, entry in files.items() if any(item["scope_kind"] == "module" for item in entry["torch"])]
    inside_functions = [
        path
        for path, entry in files.items()
        if entry["torch"] and not any(item["scope_kind"] == "module" for item in entry["torch"])
    ]
    return {
        "files": files,
        "summary": {
            "scanned": len(files),
            "torch_at_import_time": at_import,
            "neuroevolution_torch_at_import_time": [
                path for path in at_import if path.startswith("examples/neuroevolution/")
            ],
            "torch_only_inside_functions": inside_functions,
            "no_torch": [path for path, entry in files.items() if not entry["torch"]],
            "tensordict_or_genesis": [
                path for path, entry in files.items() if entry["genesis"] or entry["tensordict"] or entry["rsl_rl"]
            ],
            "rule": (
                "scope_kind == 'module' means the import runs when the module is imported; "
                "'function'/'method' means it is deferred"
            ),
        },
    }


# --- committed boundary numbers ---------------------------------------------------
def _record_ref(name: str) -> dict:
    """Load one committed record and keep the provenance fields needed to cite it."""
    path = RESULTS_DIR / name
    if not path.is_file():
        return {"source": f"results/{name}", "exists": False}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "source": f"results/{name}",
        "exists": True,
        "kind": data.get("kind"),
        "status": data.get("status"),
        "created_utc": data.get("created_utc"),
        "git_head": (data.get("git") or {}).get("head"),
        "data": data,
    }


def _replay_boundary(name: str) -> dict:
    """Extract the per-step ``state_read`` share and ms/step from a committed replay record."""
    ref = _record_ref(name)
    if not ref["exists"]:
        return ref
    data = ref.pop("data")
    groups = data["measurements"]["generations"][0]["timings"]["groups"]
    aggregate = data["timings"]["groups"]
    generation_total = groups["generation_total"]["seconds"]
    state_read = groups["state_read"]
    values = {
        "generation": 0,
        "steps": data["measurements"]["generations"][0]["steps"],
        "wall_time_s": data["measurements"]["generations"][0]["wall_time_s"],
        "state_read_s": _r(state_read["seconds"]),
        "state_read_ms_per_step": _r(state_read["per_step_s"] * 1000.0),
        "state_read_share_of_generation": _r(state_read["seconds"] / generation_total),
        "per_step_ms": {
            tag: _r(group["per_step_s"] * 1000.0) for tag, group in groups.items() if group.get("per_step_s") is not None
        },
        "aggregate_over_two_generations": {
            "state_read_ms_per_step": _r(aggregate["state_read"]["mean_s"] * 1000.0),
            "state_read_share_of_generation": _r(
                aggregate["state_read"]["seconds"] / aggregate["generation_total"]["seconds"]
            ),
        },
    }
    return {**ref, "values": values}


def _throughput_boundary(name: str) -> dict:
    """Extract the per-step timer means and the throughput block from a committed throughput record."""
    ref = _record_ref(name)
    if not ref["exists"]:
        return ref
    data = ref.pop("data")
    per_step = data["measurements"].get("per_step_means") or {}
    return {
        **ref,
        "values": {
            "per_step_ms": {
                tag: _r(slot["per_step_s"] * 1000.0) for tag, slot in per_step.items() if slot.get("per_step_s") is not None
            },
            "per_step_s": {tag: slot.get("per_step_s") for tag, slot in per_step.items()},
            "throughput": data["measurements"].get("throughput"),
            "state_read_parity": data["measurements"].get("state_read_parity"),
        },
    }


def _residency_boundary(name: str) -> dict:
    """Extract the host calls/bytes/syncs and the device-resident intermediates of a residency record."""
    ref = _record_ref(name)
    if not ref["exists"]:
        return ref
    data = ref.pop("data")
    measurements = data["measurements"]
    comparison = measurements.get("comparison") or {}
    device_side = (measurements.get("sides") or {}).get("device") or {}
    numpy_side = (measurements.get("sides") or {}).get("numpy") or {}
    counters = ("host_calls_per_step", "host_bytes_per_step", "sync_calls_per_step")
    return {
        **ref,
        "values": {
            "host_calls_per_step": comparison.get("host_calls_per_step"),
            "host_bytes_per_step": comparison.get("host_bytes_per_step"),
            "sync_calls_per_step": comparison.get("sync_calls_per_step"),
            "steps_per_s_median": comparison.get("steps_per_s_median"),
            "device_side_counters": {key: device_side.get(key) for key in counters},
            "numpy_side_counters": {key: numpy_side.get(key) for key in counters},
            "device_resident_intermediates": device_side.get("device_resident_intermediates"),
        },
    }


def measured_boundary() -> dict:
    """Return the committed boundary blocks of both pilots, each with its source and git head."""
    measured = {
        task: {
            "replay": _replay_boundary(f"env_{task}_replay_seed0.json"),
            "throughput_numpy_cpu_pop8": _throughput_boundary(f"env_{task}_throughput_numpy_cpu_pop8.json"),
            "residency_gpu": _residency_boundary(f"residency_{task}_gpu.json"),
        }
        for task in ("go2", "hover")
    }
    return {
        **measured,
        "note": (
            "each block names its source record and that record's git head; state_read is the "
            "NumPy-side device->host read of the full state, and the per-step share is relative to "
            "generation_total"
        ),
    }
