"""The task-agnostic batched-episode environment contract of the neuroevolution layer.

A trainer only needs one thing from a task: turn the four leaves of a TensorNEAT genome payload into one generation of episode returns.
:class:`BatchedEpisodeEnv` is that seam, and :func:`validate_evaluation` is the check the trainer runs on every :meth:`BatchedEpisodeEnv.evaluate` result before it feeds the evolution step.
Neither the protocol nor the validator imports evoxc, TensorNEAT, torch or genesis, so a task module can be imported and tested without a simulator.

The contract in one place: ``rows = rows_for(pop_size)`` rows are simulated per generation, policy-major, so policy ``p`` owns rows ``3p, 3p+1, 3p+2``.
Every policy of a generation is scored on the same ordered triple of episode seeds (see :mod:`seeding`), and every generation gets a fresh triple.
Rows run to ``max_episode_length`` or to their task-specific ending, whichever comes first; a row that ends early simply stops accumulating return.
A task that ends some episodes without a score reports them as non-finite, and :func:`_fitness_from_episode_returns` is the layer's single rule that counts every such episode as 0 in the float32 mean.
:data:`RESIDENCIES` names the two ways a task can satisfy the contract: ``"numpy"`` hands host arrays back, ``"device"`` hands the four :data:`DEVICE_KEYS` back as tensors on its ``device`` and is validated by :func:`validate_evaluation_device` from tensor metadata only.
"""

from __future__ import annotations

from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np

from seeding import EPISODES_PER_POLICY

__all__ = [
    "BatchedEpisodeEnv",
    "DEVICE_KEYS",
    "OPTIONAL_KEYS",
    "REQUIRED_KEYS",
    "RESIDENCIES",
    "rows_for",
    "validate_evaluation",
    "validate_evaluation_device",
]

#: The two supported residencies of an environment: host numpy arrays, or device tensors.
RESIDENCIES = ("numpy", "device")

#: Result keys that must be device-resident tensors under ``residency="device"``.
DEVICE_KEYS = ("seeds", "episode_returns", "returns", "fitness")

#: Result keys every :meth:`BatchedEpisodeEnv.evaluate` must return.
REQUIRED_KEYS = ("seeds", "episode_returns", "returns", "fitness", "steps", "env_steps", "wall_time_s")

#: Result keys a task may add on top of the required ones, with the meaning the trainer expects.
OPTIONAL_KEYS = ("steps_per_s", "env_steps_per_s", "cache_hits", "cache_misses", "end_reasons")


@runtime_checkable
class BatchedEpisodeEnv(Protocol):
    """One task's batched-episode environment: a generation of rollouts in, episode returns out.

    ``rows`` is the number of simulated rows, ``episodes_per_policy * pop_size``, and it is the length of every per-row array.
    ``episodes_per_policy`` is :data:`seeding.EPISODES_PER_POLICY` unless a task deliberately deviates.
    ``max_episode_length`` is the per-episode step cap of the task, in scene steps.
    ``cache`` is the task's cached executable registry, or None when it compiles nothing.
    ``residency`` is one of :data:`RESIDENCIES`: ``"numpy"`` (the default reference path, every result a host array) or ``"device"`` (the four :data:`DEVICE_KEYS` are tensors on ``device``).
    Device residency requires the other two attributes: ``backend`` (the ``evoxc`` backend serving ``device``) and ``device`` (an ``evoxc`` device); both are ``None``/``"numpy"`` for a numpy env.

    :meth:`evaluate` runs exactly one generation: it receives the genome payload and the generation index, and returns the rollout statistics of that generation.
    ``pop_arrays`` is ``Sequence[np.ndarray] | Mapping[str, np.ndarray]`` for a numpy env and ``Sequence[evoxc.Tensor] | Mapping[str, evoxc.Tensor]`` for a device env: the real trainers pass the four-leaf TensorNEAT payload as a tuple in ``GenomeTensors`` order, while the mapping form names the leaves.
    Under ``residency="device"`` the payload arrives as the four device tensors (``node_keys`` int32 ``(P,N)``, ``node_attrs`` float32 ``(P,N,A)``, ``conn_keys`` int32 ``(P,E,CK)``, ``conn_attrs`` float32 ``(P,E,CA)``) and is never a host array.
    Rows are policy-major, so policy ``p`` owns rows ``3p..3p+2``, and every policy of a generation gets the SAME ordered triple of episode seeds while each generation gets a fresh triple.
    The returned mapping must hold ``seeds`` (``(episodes,)`` int64, the generation's ordered seeds), ``episode_returns`` (``(pop_size, episodes)`` float32, policy-major), ``returns`` (``rows``, float32, the flattened policy-major view), ``fitness`` (``(pop_size,)`` float32, the per-policy mean with an unscored episode counted as 0), ``steps`` (episode steps taken), ``env_steps`` (``steps * rows``, or the exact simulated environment steps) and ``wall_time_s`` (wall-clock seconds of the generation).
    A device env returns those four keys as device tensors on its own device and keeps ``steps``/``env_steps``/``wall_time_s`` host values, because they are logging, never training data; :func:`validate_evaluation_device` is the validator for that residency.
    It may also hold ``steps_per_s``, ``env_steps_per_s``, ``cache_hits``, ``cache_misses`` and ``end_reasons`` (one counter per ending tag), plus arbitrary task diagnostics such as the init qpos or the commands.
    :meth:`close` releases the scene and the compiled executables; it is called once, after the last generation.
    """

    rows: int
    episodes_per_policy: int
    max_episode_length: int
    cache: Any
    residency: str
    backend: Any
    device: Any

    def evaluate(self, pop_arrays: Sequence[np.ndarray] | Mapping[str, np.ndarray], generation: int) -> dict:
        """Run one generation of rollouts and return its statistics mapping."""
        ...

    def close(self) -> None:
        """Release the scene and the executables of this environment."""
        ...


def rows_for(pop_size, episodes=EPISODES_PER_POLICY):
    """Return the number of simulated rows of a population, ``pop_size * episodes``.

    Args:
        pop_size (int): Policies per generation, non-negative.
        episodes (int): Episodes per policy, ``EPISODES_PER_POLICY`` by default.

    Returns:
        int: Rows the environment must simulate for one generation.
    """
    pop_size = int(pop_size)
    episodes = int(episodes)
    if pop_size < 0 or episodes < 1:
        raise ValueError(f"pop_size must be >= 0 and episodes >= 1, got {pop_size} and {episodes}")
    return pop_size * episodes


def _numeric(value):
    """Return ``value`` as a finite-agnostic float, or None when it is not a number."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fitness_from_episode_returns(episode_returns):
    """Return the per-policy fitness of an ``(pop_size, episodes)`` episode-return array.

    This is the layer's single aggregation rule: an episode that was not scored (NaN, +inf or -inf) counts as 0.
    The mean is taken in float32 because that is the arithmetic of the compiled ``fitness_graph`` (``evoxc.reduce_mean``), and a float64 mean differs from that graph by more than the ``1e-6`` tolerance at episode-return magnitudes of ~100 (the Go2 pilot).
    :func:`validate_evaluation` and ``trainer._aggregate`` both call this helper, so the validator and the trainer cannot disagree about what an env's fitness must be.

    Args:
        episode_returns (array_like): Episode returns, ``(pop_size, episodes)``, policy-major.

    Returns:
        numpy.ndarray: Fitness, ``(pop_size,)`` float32, the float32 mean of each row with every non-finite entry counted as 0.
    """
    episode_returns = np.asarray(episode_returns, dtype=np.float32)
    return np.nan_to_num(episode_returns, nan=0.0, posinf=0.0, neginf=0.0).mean(axis=1).astype(np.float32)


def validate_evaluation(result, pop_size, episodes=EPISODES_PER_POLICY):
    """Raise one ``ValueError`` listing every problem of an ``evaluate`` result, or return None.

    The check is deliberately total: it collects every violation instead of failing on the first one, so a broken task reports its whole contract violation in one message.
    An unscored episode is allowed to be non-finite: the row-layout check matches NaN to NaN, and fitness must count every non-finite episode (NaN, +inf or -inf) as 0 in the float32 mean that :func:`_fitness_from_episode_returns` owns.

    Args:
        result (Mapping): The mapping returned by :meth:`BatchedEpisodeEnv.evaluate`.
        pop_size (int): Policies the generation was run for.
        episodes (int): Episodes per policy, ``EPISODES_PER_POLICY`` by default.

    Returns:
        None: The result is well formed.
    """
    pop_size = int(pop_size)
    episodes = int(episodes)
    rows = rows_for(pop_size, episodes)
    problems = []
    if not isinstance(result, Mapping):
        raise ValueError(f"evaluate must return a mapping, got {type(result).__name__}")
    for key in REQUIRED_KEYS:
        if key not in result:
            problems.append(f"missing required key {key!r}")

    if "seeds" in result:
        seeds = np.asarray(result["seeds"])
        if not np.issubdtype(seeds.dtype, np.integer):
            problems.append(f"seeds must have an integer dtype, got {seeds.dtype}")
        elif seeds.shape != (episodes,):
            problems.append(f"seeds must have shape ({episodes},), got {seeds.shape}")
        elif np.unique(seeds).size != episodes:
            problems.append(f"seeds must be pairwise distinct, got {seeds.tolist()}")

    episode_returns = None
    if "episode_returns" in result:
        episode_returns = np.asarray(result["episode_returns"])
        if not np.issubdtype(episode_returns.dtype, np.floating):
            problems.append(f"episode_returns must have a floating dtype, got {episode_returns.dtype}")
            episode_returns = None
        elif episode_returns.shape != (pop_size, episodes):
            problems.append(f"episode_returns must have shape ({pop_size}, {episodes}), got {episode_returns.shape}")
            episode_returns = None

    if "returns" in result:
        returns = np.asarray(result["returns"])
        if not np.issubdtype(returns.dtype, np.floating):
            problems.append(f"returns must have a floating dtype, got {returns.dtype}")
        elif returns.shape != (rows,):
            problems.append(f"returns must have shape ({rows},), got {returns.shape}")
        elif episode_returns is not None and not np.allclose(
            returns.reshape(pop_size, episodes), episode_returns, atol=1e-6, rtol=0, equal_nan=True
        ):
            problems.append(
                "returns must be the policy-major flattening of episode_returns, i.e. "
                "returns.reshape(pop_size, episodes) == episode_returns (NaN matching NaN)"
            )

    if "fitness" in result:
        fitness = np.asarray(result["fitness"])
        if not np.issubdtype(fitness.dtype, np.floating):
            problems.append(f"fitness must have a floating dtype, got {fitness.dtype}")
        elif fitness.shape != (pop_size,):
            problems.append(f"fitness must have shape ({pop_size},), got {fitness.shape}")
        elif episode_returns is not None:
            expected = _fitness_from_episode_returns(episode_returns)
            if not np.allclose(fitness, expected, atol=1e-6, rtol=0):
                problems.append(
                    "fitness must be the episode-mean of episode_returns with every non-finite episode (NaN, +inf "
                    f"or -inf) counted as 0, got {fitness.tolist()} instead of {expected.tolist()}"
                )

    for key in ("steps", "env_steps", "wall_time_s"):
        if key not in result:
            continue
        value = _numeric(result[key])
        if value is None:
            problems.append(f"{key} must be a number, got {result[key]!r}")
        elif not value >= 0.0:
            problems.append(f"{key} must be non-negative, got {value!r}")

    if problems:
        raise ValueError("invalid evaluate result:\n  - " + "\n  - ".join(problems))


def _is_device_tensor(value):
    """Return True when ``value`` is a device tensor (duck-typed, never materialized)."""
    if isinstance(value, np.ndarray):
        return False
    return all(hasattr(value, attribute) for attribute in ("shape", "dtype", "device"))


def _placement_label(device):
    """``"kind:index"`` of a device object, or its ``str`` when it is not an evoxc device."""
    kind = getattr(device, "kind", None)
    if kind is None:
        return str(device)
    return f"{kind}:{getattr(device, 'index', 0)}"


def _device_tensor_problem(value, key, dtype_predicate, dtype_label, shape, device, name):
    """Return a list with every problem of one device-resident result key (metadata only)."""
    if not _is_device_tensor(value):
        return [
            f"{key} must be a device tensor (an evoxc tensor with .shape/.dtype/.device) under "
            f"residency='device', got {type(value).__name__}"
        ]
    problems = []
    dtype = np.dtype(value.dtype)
    if not dtype_predicate(dtype):
        problems.append(f"{key} must have {dtype_label} dtype, got {value.dtype}")
    if tuple(int(dim) for dim in value.shape) != shape:
        problems.append(f"{key} must have shape {shape}, got {tuple(int(dim) for dim in value.shape)}")
    if device is not None:
        actual = _placement_label(value.device)
        target = _placement_label(device)
        if actual != target:
            problems.append(f"{key} must live on {target} ({name}), got a tensor on {actual}")
    return problems


def validate_evaluation_device(result, pop_size, episodes=EPISODES_PER_POLICY, device=None):
    """Raise one ``ValueError`` listing every problem of a device-resident ``evaluate`` result.

    This is the ``residency="device"`` validator: it checks the mapping's keys, dtypes, shapes and device placement from tensor metadata only, and it never materializes a tensor.
    The arithmetic of the four :data:`DEVICE_KEYS` is NOT checked here -- ``returns`` tiling and the ``fitness`` rule are verified in-graph by ``evolution_graphs.build_verification``, because materializing them to compare on the host is exactly what device residency avoids.
    Like :func:`validate_evaluation` the check is total: it collects every violation instead of failing on the first one.
    The device keys must be ``evoxc`` tensors (anything with ``.shape``/``.dtype``/``.device``); a numpy array is rejected, since a device result carries no host training data.
    ``steps``, ``env_steps`` and ``wall_time_s`` must be host non-negative numbers and ``end_reasons`` a mapping of int counts; those are logging, never training data.

    Args:
        result (Mapping): The mapping returned by :meth:`BatchedEpisodeEnv.evaluate` under device residency.
        pop_size (int): Policies the generation was run for.
        episodes (int): Episodes per policy, ``EPISODES_PER_POLICY`` by default.
        device (evoxc.Device | None): The expected placement; ``None`` accepts any device.

    Returns:
        None: The device result is well formed.
    """
    pop_size = int(pop_size)
    episodes = int(episodes)
    rows = rows_for(pop_size, episodes)
    if not isinstance(result, Mapping):
        raise ValueError(f"evaluate must return a mapping, got {type(result).__name__}")
    problems = []
    for key in REQUIRED_KEYS:
        if key not in result:
            problems.append(f"missing required key {key!r}")

    name = "residency='device'"
    if "seeds" in result:
        problems += _device_tensor_problem(
            result["seeds"],
            "seeds",
            lambda dtype: np.issubdtype(dtype, np.integer),
            "an integer",
            (episodes,),
            device,
            name,
        )
    if "episode_returns" in result:
        problems += _device_tensor_problem(
            result["episode_returns"],
            "episode_returns",
            lambda dtype: np.issubdtype(dtype, np.floating),
            "a floating",
            (pop_size, episodes),
            device,
            name,
        )
    if "returns" in result:
        problems += _device_tensor_problem(
            result["returns"],
            "returns",
            lambda dtype: np.issubdtype(dtype, np.floating),
            "a floating",
            (rows,),
            device,
            name,
        )
    if "fitness" in result:
        problems += _device_tensor_problem(
            result["fitness"],
            "fitness",
            lambda dtype: np.issubdtype(dtype, np.floating),
            "a floating",
            (pop_size,),
            device,
            name,
        )

    for key in ("steps", "env_steps", "wall_time_s"):
        if key not in result:
            continue
        if _is_device_tensor(result[key]):
            problems.append(f"{key} must be a host value under residency='device', got a device tensor")
            continue
        value = _numeric(result[key])
        if value is None:
            problems.append(f"{key} must be a number, got {result[key]!r}")
        elif not value >= 0.0:
            problems.append(f"{key} must be non-negative, got {value!r}")

    if "end_reasons" in result:
        reasons = result["end_reasons"]
        if not isinstance(reasons, Mapping):
            problems.append(f"end_reasons must be a mapping of counts, got {type(reasons).__name__}")
        else:
            for reason, count in reasons.items():
                if isinstance(count, bool) or not isinstance(count, (int, np.integer)):
                    problems.append(f"end_reasons[{reason!r}] must be an int count, got {count!r}")

    if problems:
        raise ValueError("invalid device evaluate result:\n  - " + "\n  - ".join(problems))
