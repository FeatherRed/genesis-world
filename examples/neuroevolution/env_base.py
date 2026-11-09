"""The task-agnostic batched-episode environment contract of the neuroevolution layer.

A trainer only needs one thing from a task: turn the four leaves of a TensorNEAT genome payload into one generation of episode returns.
:class:`BatchedEpisodeEnv` is that seam, and :func:`validate_evaluation` is the check the trainer runs on every :meth:`BatchedEpisodeEnv.evaluate` result before it feeds the evolution step.
Neither the protocol nor the validator imports evoxc, TensorNEAT, torch or genesis, so a task module can be imported and tested without a simulator.

The contract in one place: ``rows = rows_for(pop_size)`` rows are simulated per generation, policy-major, so policy ``p`` owns rows ``3p, 3p+1, 3p+2``.
Every policy of a generation is scored on the same ordered triple of episode seeds (see :mod:`seeding`), and every generation gets a fresh triple.
Rows run to ``max_episode_length`` or to their task-specific ending, whichever comes first; a row that ends early simply stops accumulating return.
A task that ends some episodes without a score reports them as non-finite and counts them as 0 when it aggregates fitness.
"""

from __future__ import annotations

from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np

from seeding import EPISODES_PER_POLICY

__all__ = [
    "BatchedEpisodeEnv",
    "OPTIONAL_KEYS",
    "REQUIRED_KEYS",
    "rows_for",
    "validate_evaluation",
]

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

    :meth:`evaluate` runs exactly one generation: it receives the genome payload and the generation index, and returns the rollout statistics of that generation.
    ``pop_arrays`` is ``Sequence[np.ndarray] | Mapping[str, np.ndarray]``: the real trainers pass the four-leaf TensorNEAT payload as a tuple in ``GenomeTensors`` order, while the mapping form names the leaves.
    Rows are policy-major, so policy ``p`` owns rows ``3p..3p+2``, and every policy of a generation gets the SAME ordered triple of episode seeds while each generation gets a fresh triple.
    The returned mapping must hold ``seeds`` (``(episodes,)`` int64, the generation's ordered seeds), ``episode_returns`` (``(pop_size, episodes)`` float32, policy-major), ``returns`` (``rows``, float32, the flattened policy-major view), ``fitness`` (``(pop_size,)`` float32, the per-policy mean with an unscored episode counted as 0), ``steps`` (episode steps taken), ``env_steps`` (``steps * rows``, or the exact simulated environment steps) and ``wall_time_s`` (wall-clock seconds of the generation).
    It may also hold ``steps_per_s``, ``env_steps_per_s``, ``cache_hits``, ``cache_misses`` and ``end_reasons`` (one counter per ending tag), plus arbitrary task diagnostics such as the init qpos or the commands.
    :meth:`close` releases the scene and the compiled executables; it is called once, after the last generation.
    """

    rows: int
    episodes_per_policy: int
    max_episode_length: int
    cache: Any

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


def validate_evaluation(result, pop_size, episodes=EPISODES_PER_POLICY):
    """Raise one ``ValueError`` listing every problem of an ``evaluate`` result, or return None.

    The check is deliberately total: it collects every violation instead of failing on the first one, so a broken task reports its whole contract violation in one message.
    An unscored episode is allowed to be non-finite: the row-layout check matches NaN to NaN, and fitness must count it as 0.

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
            expected = np.nan_to_num(episode_returns, nan=0.0).mean(axis=1)
            if not np.allclose(fitness, expected, atol=1e-6, rtol=0):
                problems.append(
                    "fitness must be the episode-mean of episode_returns with an unscored episode counted as 0, "
                    f"got {fitness.tolist()} instead of {expected.tolist()}"
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
