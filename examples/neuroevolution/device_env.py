"""A simulator-free, device-resident environment: the plumbing check of residency contract v2.

:class:`SyntheticDeviceEnv` is the environment the device self-check and the hardware
verification CLI run, and it exists so device residency can be exercised without a
scene, a GPU physics engine or a rollout loop.
Its returns are a deterministic function of the genome, produced in one compiled graph
that consumes the four payload leaves exactly as a pilot's policy graph would, so a
placement, staging or tiling mistake fails here in a fraction of a second instead of
after a simulation.

The graph
---------
``returns_graph`` takes the four ``GenomeTensors`` leaves and returns three tensors,
the same three the interface contract names:

* ``episode_returns (pop_size, episodes)`` float32 -- per-policy return plus a fixed
  per-episode offset, so the values differ across policies *and* across episodes and
  an evolution step actually changes the fitness;
* ``returns (rows,)`` float32 -- the policy-major flattening of ``episode_returns``;
* ``fitness (pop_size,)`` float32 -- :func:`evolution_graphs.fitness_rule_graph` of
  ``returns``, i.e. the layer's one aggregation rule computed on the device.

The per-policy value is ``NODE_SCALE * sum(node_attrs) + CONN_SCALE * sum(|conn_attrs|)``,
both reductions over the whole leaf, which is genome-dependent from generation 0.
``nan_episodes=(i, ...)`` multiplies those episode columns by NaN in-graph, which is how
the "an unscored episode counts as 0" rule is exercised on a device result.

Staging
-------
Device residency transfers exactly one thing: the generation's ``(episodes,)`` int64 seed
triple, 24 bytes of metadata, moved to the device in :meth:`evaluate`.
Everything else is produced on the device and stays there; the host metrics
(``steps``, ``env_steps``, ``wall_time_s``, ``steps_per_s``, ``env_steps_per_s``,
``end_reasons``, the cache counters) are plain Python values because they are logging,
never training data.

``residency="numpy"`` runs the very same graph with host arrays in and host arrays out,
which is the parity reference for a device env built on the same backend/device.

A cpu-kind device is accepted: it is a plumbing-only mode (placement, tiling and staging
are exercised, the accelerator is not), and the pilots' envs refuse it for that reason.

Nothing here imports ``torch``, ``genesis`` or ``tensordict``; and unlike the pilots this
module needs no ``tensorneat`` either -- it only speaks ``evoxc``.
"""

from __future__ import annotations

import time

import numpy as np

import evoxc

from device_runtime import assert_device_resident, is_device_tensor
from env_base import (
    RESIDENCIES,
    rows_for,
    validate_evaluation,
    validate_evaluation_device,
)
from evolution_graphs import ExecutableCache, _fitness_rule_ops
from evox_runtime import (
    _attach,
    _contiguous,
    _device_name,
    _outputs,
    _pop_specs,
    _to_numpy,
    resolve_backend_device,
)
from seeding import EPISODES_PER_POLICY, generation_seed_table

__all__ = ["SyntheticDeviceEnv", "NODE_SCALE", "CONN_SCALE", "EPISODE_OFFSETS"]

#: Weight of the node-attribute sum in the synthetic per-policy return.
NODE_SCALE = 1.25

#: Weight of the absolute connection-attribute sum in the synthetic per-policy return.
CONN_SCALE = 0.05

#: Fixed per-episode offsets added to a policy's return, so episodes differ by construction (cycled past 3 episodes).
EPISODE_OFFSETS = (0.0, 0.5, -0.25)


def _zero_nonfinite(value):
    """Emit ``nan_to_num``-equivalent ops: a non-finite float32 fails ``|x| <= FLT_MAX``.

    ``evoxc.nan_to_num`` is not lowerable by the IREE stablehlo exporter, so the sanitize
    is spelled as one comparison plus a select (see ``evolution_graphs._fitness_rule_ops``).
    """
    return evoxc.select(
        evoxc.less_equal(evoxc.abs(value), float(np.finfo(np.float32).max)), value, 0.0
    )


def _returns_ops(node_attrs, conn_attrs, n_policies, episodes):
    """Emit the synthetic returns ops: (episode_returns, returns, fitness).

    The attribute leaves are sanitized first: an unused node slot of a NEAT payload holds
    NaN, so a raw reduction would make every policy's return NaN and the fitness useless.
    """
    nodes = _zero_nonfinite(node_attrs)
    conns = _zero_nonfinite(conn_attrs)
    per_policy = evoxc.add(
        evoxc.multiply(evoxc.reduce_sum(nodes, axes=(1, 2)), NODE_SCALE),
        evoxc.multiply(evoxc.reduce_sum(evoxc.abs(conns), axes=(1, 2)), CONN_SCALE),
    )
    offsets = np.resize(np.asarray(EPISODE_OFFSETS, dtype=np.float32), int(episodes))
    lifted = evoxc.reshape(per_policy, (n_policies, 1))
    offset = evoxc.constant(evoxc.from_numpy(offsets.reshape(1, int(episodes))))
    episode_returns = evoxc.add(lifted, offset)
    returns = evoxc.reshape(episode_returns, (n_policies * int(episodes),))
    return episode_returns, returns, _fitness_rule_ops(returns, n_policies, int(episodes))


@evoxc.defn
def returns_graph(node_keys, node_attrs, conn_keys, conn_attrs, n_policies, episodes):
    """Four payload leaves -> ``episode_returns``, ``returns`` and the rule-based ``fitness``."""
    del node_keys, conn_keys  # the synthetic return reads the two attribute leaves only
    return list(_returns_ops(node_attrs, conn_attrs, n_policies, episodes))


def build_returns_graph(pop_arrays, *, n_policies, episodes_per_policy, backend="numpy", device=None, nan_episodes=()):
    """Compile the synthetic returns graph for one payload shape.

    Args:
        pop_arrays: The four genome payload leaves (host numpy or device tensors).
        n_policies (int): Population size.
        episodes_per_policy (int): Episodes per policy.
        backend (str): ``evoxc`` backend name.
        device (str | evoxc.Device | None): ``evoxc`` device.
        nan_episodes (tuple[int, ...]): Episode columns to turn into NaN in-graph.

    Returns:
        evoxc.Executable: bound executable; :meth:`SyntheticDeviceEnv.evaluate` runs it.
    """
    indices = tuple(int(index) for index in nan_episodes)
    for index in indices:
        if not 0 <= index < int(episodes_per_policy):
            raise ValueError(f"nan_episodes index {index} is outside [0, {episodes_per_policy})")
    resolved_backend, resolved_device = resolve_backend_device(backend, device)
    rows = rows_for(n_policies, episodes_per_policy)
    kernel = returns_graph
    if indices:
        mask = np.ones((1, int(episodes_per_policy)), dtype=np.float32)
        mask[0, list(indices)] = np.nan

        def kernel(node_keys, node_attrs, conn_keys, conn_attrs, n_policies, episodes):
            episode_returns, _, _ = _returns_ops(node_attrs, conn_attrs, n_policies, episodes)
            masked = evoxc.multiply(episode_returns, evoxc.constant(evoxc.from_numpy(mask)))
            flat = evoxc.reshape(masked, (n_policies * int(episodes),))
            return [masked, flat, _fitness_rule_ops(flat, n_policies, int(episodes))]

    executable = evoxc.build(
        kernel,
        *_pop_specs(pop_arrays),
        n_policies,
        int(episodes_per_policy),
        backend=resolved_backend,
        device=resolved_device,
    )
    return _attach(
        executable,
        graph="synthetic_returns",
        rows=rows,
        n_policies=int(n_policies),
        episodes_per_policy=int(episodes_per_policy),
        nan_episodes=indices,
        device_name=_device_name(resolved_device),
    )


class SyntheticDeviceEnv:
    """A simulator-free env implementing the v2 contract, in either residency.

    Args:
        pop_size (int): Policies per generation, hence ``pop_size * episodes`` rows.
        num_inputs (int): Observation width the payload shape is sized for (45 for Go2).
        num_outputs (int): Action width the payload shape is sized for (12 for Go2).
        episodes (int): Episodes per policy, ``EPISODES_PER_POLICY`` by default.
        backend (str | None): ``evoxc`` backend; ``None`` resolves like the graph modules do.
        device (evoxc.Device | str | None): ``evoxc`` device; a cpu-kind device is a
            plumbing-only mode (placement/staging are exercised, no accelerator is used).
        residency (str): ``"device"`` (the default: the four :data:`env_base.DEVICE_KEYS`
            come back as device tensors) or ``"numpy"`` (host arrays out, the parity reference).
        seed (int): Master seed of the env's own episode-seed stream.
        horizon (int): Simulated steps per episode, reported as ``steps``.
        cache (ExecutableCache | None): Cache the returns graph is built through.
        nan_episodes (tuple[int, ...]): Episode columns reported as NaN (an unscored episode).
        seed_table (array-like | None): Replaces the table :func:`seeding.generation_seed_table` draws.

    Attributes:
        rows (int): ``pop_size * episodes``.
        episodes_per_policy (int): Episodes per policy.
        max_episode_length (int): The episode horizon, in steps.
        cache (ExecutableCache): The executable cache this env builds through.
        backend (str): The resolved ``evoxc`` backend.
        device (evoxc.Device): The resolved device.
        residency (str): ``"numpy"`` or ``"device"``.
    """

    def __init__(
        self,
        pop_size,
        *,
        num_inputs=45,
        num_outputs=12,
        episodes=EPISODES_PER_POLICY,
        backend=None,
        device=None,
        residency="device",
        seed=0,
        horizon=25,
        cache=None,
        nan_episodes=(),
        seed_table=None,
    ):
        if residency not in RESIDENCIES:
            raise ValueError(f"residency must be one of {RESIDENCIES}, got {residency!r}")
        self.pop_size = int(pop_size)
        if self.pop_size < 1:
            raise ValueError(f"pop_size must be positive, got {pop_size!r}")
        self.num_inputs = int(num_inputs)
        self.num_outputs = int(num_outputs)
        self.episodes_per_policy = int(episodes)
        self.rows = rows_for(self.pop_size, self.episodes_per_policy)
        self.max_episode_length = int(horizon)
        if self.max_episode_length < 1:
            raise ValueError(f"horizon must be positive, got {horizon!r}")
        self.residency = residency
        self.seed = int(seed)
        self.nan_episodes = tuple(int(index) for index in nan_episodes)
        self.backend, self.device = resolve_backend_device(backend, device)
        self.device_name = _device_name(self.device)
        self.cache = ExecutableCache() if cache is None else cache
        self._owns_cache = cache is None
        self._executable = None
        self._closed = False
        self.seed_table = (
            generation_seed_table(self.seed, 8, episodes=self.episodes_per_policy)
            if seed_table is None
            else np.asarray(seed_table, dtype=np.int64)
        )

    # --- Shape ------------------------------------------------------------------------

    def _payload_placeholders(self):
        """Stand-in payload leaves of the ``DefaultGenome`` widths, for the graph's specs only.

        ``DefaultGenome(max_nodes=80, max_conns=640)`` is what the trainers build, so these
        are the shapes and dtypes the compiled graph is traced against; ``evaluate`` checks
        the real leaves through ``evoxc``'s own run-boundary checks.
        """
        nodes, conns = 80, 640
        return (
            np.zeros((self.pop_size, nodes), dtype=np.int32),
            np.zeros((self.pop_size, nodes, 4), dtype=np.float32),
            np.zeros((self.pop_size, conns, 2), dtype=np.int32),
            np.zeros((self.pop_size, conns, 1), dtype=np.float32),
        )

    def _returns_key(self):
        """The cache key of the returns graph."""
        return (
            "synthetic_returns",
            self.pop_size,
            self.episodes_per_policy,
            self.nan_episodes,
            self.backend,
            self.device_name,
        )

    # --- Generations ------------------------------------------------------------------

    def evaluate(self, pop_arrays, generation):
        """Run one synthetic generation and return its statistics mapping.

        Args:
            pop_arrays: The four genome payload leaves -- device tensors under
                ``residency="device"`` (asserted), host arrays under ``residency="numpy"``.
            generation (int): The generation index; selects the episode-seed triple.

        Returns:
            dict: The four :data:`env_base.DEVICE_KEYS` plus the host metrics.
        """
        start = time.perf_counter()
        executable = self.executables()
        leaves = tuple(pop_arrays)
        names = ("node_keys", "node_attrs", "conn_keys", "conn_attrs")
        if len(leaves) != len(names):
            raise ValueError(f"pop_arrays must hold the {len(names)} GenomeTensors payloads, got {len(leaves)}")
        device_resident = self.residency == "device"
        for value, name in zip(leaves, names, strict=True):
            if device_resident:
                assert_device_resident(value, self.device, name)
            elif is_device_tensor(value):
                raise TypeError(f"{name} must be a host array under residency='numpy', got {type(value).__name__}")
        operands = (
            leaves
            if device_resident
            else tuple(_from_host(value, self.device, self.backend) for value in leaves)
        )
        flat = _outputs(evoxc.run(executable, *operands, self.pop_size, self.episodes_per_policy))
        result = self._result(flat, generation, start)
        if device_resident:
            validate_evaluation_device(result, self.pop_size, self.episodes_per_policy, self.device)
        else:
            validate_evaluation(result, self.pop_size, self.episodes_per_policy)
        return result

    def _result(self, flat, generation, start):
        """Assemble the generation's mapping from the graph's three outputs and the host metrics."""
        seeds = self.seed_table[int(generation) % len(self.seed_table)]
        steps = self.max_episode_length
        wall_time_s = time.perf_counter() - start
        host = {
            "steps": steps,
            "env_steps": steps * self.rows,
            "wall_time_s": wall_time_s,
            "steps_per_s": steps / wall_time_s if wall_time_s > 0 else 0.0,
            "env_steps_per_s": (steps * self.rows) / wall_time_s if wall_time_s > 0 else 0.0,
            "end_reasons": {"timeout": self.rows},
            "cache_hits": int(getattr(self.cache, "hits", 0)),
            "cache_misses": int(getattr(self.cache, "misses", 0)),
        }
        if self.residency == "device":
            # The ONE host -> device transfer of the generation: 24 bytes of seed metadata.
            return {
                "seeds": _from_host(seeds, self.device, self.backend),
                "episode_returns": flat[0],
                "returns": flat[1],
                "fitness": flat[2],
                **host,
            }
        return {
            "seeds": np.asarray(seeds, dtype=np.int64),
            "episode_returns": np.asarray(_to_numpy(flat[0]), dtype=np.float32),
            "returns": np.asarray(_to_numpy(flat[1]), dtype=np.float32),
            "fitness": np.asarray(_to_numpy(flat[2]), dtype=np.float32),
            **host,
        }

    # --- Executables / lifecycle -------------------------------------------------------

    def executables(self):
        """Return the cached returns graph, compiling it on the first call."""
        if self._closed:
            raise RuntimeError("SyntheticDeviceEnv is closed; create a new env instead of reusing this one")
        if self._executable is None:
            placeholders = self._payload_placeholders()
            self._executable = self.cache.get(
                self._returns_key(),
                lambda: build_returns_graph(
                    placeholders,
                    n_policies=self.pop_size,
                    episodes_per_policy=self.episodes_per_policy,
                    backend=self.backend,
                    device=self.device,
                    nan_episodes=self.nan_episodes,
                ),
            )
        return self._executable

    def close(self):
        """Drop the env's own cache entry; a cache handed in by the caller is left alone."""
        if self._closed:
            return
        self._closed = True
        self._executable = None
        if self._owns_cache:
            self.cache.clear()

    def __enter__(self):
        """Return ``self`` so the env can be used as a context manager."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Release the env's own cache; a caller-owned cache is left open."""
        self.close()
        return False


def _from_host(value, device, backend):
    """Stage one host array onto ``device`` (re-selecting the backend first), always as a tensor."""
    array = _contiguous(np.asarray(value))
    if device.kind != "cpu":
        evoxc.backends.get(backend)
    return evoxc.from_numpy(array).to(device)
