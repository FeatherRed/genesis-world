"""Task-agnostic evolution-side ``evoxc`` graphs for the TensorNEAT trainers.

These graphs live here, plus the shape-keyed executable cache the trainers use so a
generation only ever builds once:

``fitness_graph``
    Episode returns ``(n_policies * episodes_per_policy,)`` float32 -> per-policy
    fitness ``(n_policies,)`` float32, the mean over each policy's episodes.

``fitness_rule_graph``
    The device transcription of ``env_base._fitness_from_episode_returns``: the
    same mean, but every non-finite episode zeroed first, so a device-resident
    fitness can be produced and checked without touching the host.

``verification_graph``
    The three scalars device residency reads back per generation: whether
    ``returns`` is the policy-major tiling of ``episode_returns``, whether
    ``fitness`` is the fitness rule applied to it, and the largest absolute
    fitness deviation.

``tell_graph``
    One TensorNEAT ``tell`` transition: ``randkey``, the four genome payload
    arrays, ``generation``, ``fitness`` and the ``SpeciesState`` leaves in, the
    flat list ``[randkey, generation, *pop, *species]`` out.

No graph knows a task: no observation width, action width, reward or
simulator constant appears here, so the Go2 walking (45 -> 12) and the drone
hover (17 -> 4) trainers compile the very same code.

Residency
---------
Every ``run_*`` entry point takes a keyword-only ``residency`` switch.
``"numpy"`` (the default) stages host arrays onto the executable's device and
materializes results back to numpy; ``"device"`` stages nothing -- operands must
already be device tensors on the executable's device (``_device_operand``) and
results are handed back as device tensors.
Only :func:`run_verification` reads scalars back in device residency.

Contracts the caller must honour
--------------------------------
* Rows are **policy-major**: rows ``episodes*p .. episodes*p+episodes-1`` are the
  ``episodes`` episodes of policy ``p``, which is the tiling
  ``algo.prepare_forward(samples=episodes)`` declares for the policy graph.
* ``build_tell`` traces with a real ``algo`` and ``config``: ``algo.tell`` is a
  trace-time operation and cannot run outside a build.
* The ``SpeciesState`` pytree layout is captured during that trace and stashed on
  the executable, so ``run_tell`` can rebuild the species tree from flat leaves
  without a ``state`` argument.

Only ``numpy``, ``evoxc`` and ``TensorNEAT`` are imported.
"""

from __future__ import annotations

import logging

import numpy as np

import evoxc

from tensorneat.algorithm.neat.species import SpeciesState
from tensorneat.genome.utils import GenomeTensors

from env_base import RESIDENCIES
from evox_runtime import (
    POP_OPERANDS,
    _attach,
    _contiguous,
    _device_name,
    _device_operand,
    _host,
    _outputs,
    _pop_specs,
    _positive_int,
    _rows_of,
    _shape_of,
    _spec_for,
    _staged,
    _to_device_array,
    _to_numpy,
    resolve_backend_device,
)

logger = logging.getLogger(__name__)


def _residency(value) -> str:
    """Validate a ``residency`` switch (``"numpy"`` or ``"device"``)."""
    if value not in RESIDENCIES:
        raise ValueError(f"residency must be one of {RESIDENCIES}, got {value!r}")
    return value


def _device_tensor(executable, value, name, shape=None, *, dtype=None):
    """Check one device-resident operand's placement, plus its shape/dtype when given, from metadata.

    ``dtype`` is ``None`` (any numeric leaf, e.g. a genome payload), ``"integer"`` or
    ``"floating"``.
    """
    operand = _device_operand(value, executable.device, getattr(executable, "backend", "numpy"), name)
    actual = np.dtype(operand.dtype)
    if dtype == "integer" and not np.issubdtype(actual, np.integer):
        raise ValueError(f"{name} must have an integer dtype, got {operand.dtype}")
    if dtype == "floating" and not np.issubdtype(actual, np.floating):
        raise ValueError(f"{name} must have a floating dtype, got {operand.dtype}")
    if shape is not None and tuple(int(dim) for dim in operand.shape) != tuple(shape):
        raise ValueError(f"{name} must have shape {tuple(shape)}, got {tuple(int(dim) for dim in operand.shape)}")
    return operand


def _staged_float_shape(executable, name: str, value, shape):
    """Stage one host float32 argument, checking its exact shape against the compiled graph."""
    arr = np.asarray(value, dtype=np.float32)
    if arr.shape != tuple(shape):
        raise ValueError(f"{name} must have shape {tuple(shape)} to match the compiled graph, got {arr.shape}")
    return _to_device_array(executable, np.ascontiguousarray(arr))


# --- Fitness ------------------------------------------------------------------------


@evoxc.defn
def fitness_graph(returns, n_policies, episodes_per_policy):
    """Episode returns ``(NP * episodes_per_policy,)`` -> fitness ``(NP,)`` float32."""
    return evoxc.reduce_mean(
        evoxc.reshape(returns, (n_policies, episodes_per_policy)),
        axes=1,
    )


def build_fitness(
    n_policies,
    *,
    episodes_per_policy=3,
    backend="numpy",
    device=None,
):
    """Compile the fitness aggregation graph (mean of one policy's episode returns).

    Args:
        n_policies (int): Population size (``algo.pop_size``).
        episodes_per_policy (int): Episodes per policy.
        backend (str): ``evoxc`` backend name.
        device (str | evoxc.Device | None): ``evoxc`` device.

    Returns:
        evoxc.Executable: bound executable; ``run_fitness`` stages its operand.
    """
    n_policies = _positive_int(n_policies, "n_policies")
    episodes_per_policy = _positive_int(episodes_per_policy, "episodes_per_policy")
    resolved_backend, resolved_device = resolve_backend_device(backend, device)
    rows = n_policies * episodes_per_policy
    executable = evoxc.build(
        fitness_graph,
        evoxc.TensorSpec((rows,), evoxc.float32, name="returns"),
        n_policies,
        episodes_per_policy,
        backend=resolved_backend,
        device=resolved_device,
    )
    return _attach(
        executable,
        graph="fitness",
        rows=rows,
        n_policies=n_policies,
        episodes_per_policy=episodes_per_policy,
        device_name=_device_name(resolved_device),
    )


def run_fitness(executable, returns, *, residency="numpy"):
    """Run the fitness graph; ``returns`` is ``(NP * episodes,)``, result ``(NP,)``.

    Args:
        executable: the bound graph from :func:`build_fitness`.
        returns: ``(NP * episodes,)`` float32 -- a host array in ``"numpy"`` residency, a
            tensor already on the executable's device in ``"device"`` residency.
        residency (str): ``"numpy"`` stages and materializes; ``"device"`` stages nothing
            and returns the device tensor.

    Returns:
        numpy.ndarray | evoxc.Tensor: ``(NP,)`` float32.
    """
    if _residency(residency) == "device":
        rows = _rows_of(executable)
        operand = _device_tensor(executable, returns, "returns", (rows,), dtype="floating")
        return _outputs(
            evoxc.run(executable, operand, executable.n_policies, executable.episodes_per_policy)
        )[0]
    outputs = evoxc.run(
        executable,
        _to_device_array(executable, _staged(executable, "returns", returns)),
        executable.n_policies,
        executable.episodes_per_policy,
    )
    return np.asarray(_to_numpy(_outputs(outputs)[0]), dtype=np.float32)


# --- Fitness rule (the device transcription of env_base's rule) ----------------------


#: Largest finite float32, the threshold that separates a scored episode from a non-finite one.
_MAX_FINITE = float(np.finfo(np.float32).max)


def _fitness_rule_ops(returns, n_policies, episodes_per_policy):
    """Emit the fitness-rule ops: reshape, zero every non-finite episode, mean over episodes.

    ``nan_to_num(x, nan=0, posinf=0, neginf=0)`` is spelled as one comparison plus a select:
    every non-finite float32 value -- NaN and both infinities -- fails ``|x| <= FLT_MAX``,
    and ``evoxc.nan_to_num`` cannot be lowered by the IREE stablehlo exporter (it raises
    ``BackendError: op 'nan_to_num' ... is not supported in v1``), while ``select`` can.
    The two agree on every float32 value, so the rule stays bit-identical to
    ``env_base._fitness_from_episode_returns``.
    """
    reshaped = evoxc.reshape(returns, (n_policies, episodes_per_policy))
    finite = evoxc.less_equal(evoxc.abs(reshaped), _MAX_FINITE)
    sanitized = evoxc.select(finite, reshaped, 0.0)
    return evoxc.reduce_mean(sanitized, axes=1)


@evoxc.defn
def fitness_rule_graph(returns, n_policies, episodes_per_policy):
    """``env_base._fitness_from_episode_returns`` as a graph: non-finite -> 0, then the float32 mean."""
    return _fitness_rule_ops(returns, n_policies, episodes_per_policy)


def build_fitness_rule(
    n_policies,
    *,
    episodes_per_policy=3,
    backend="numpy",
    device=None,
):
    """Compile the fitness-rule graph (non-finite episodes counted as 0, float32 mean).

    The rule has one host owner (``env_base._fitness_from_episode_returns``) and this
    one device transcription; the two are guarded against each other by the node's
    self-check.

    Args:
        n_policies (int): Population size.
        episodes_per_policy (int): Episodes per policy.
        backend (str): ``evoxc`` backend name.
        device (str | evoxc.Device | None): ``evoxc`` device.

    Returns:
        evoxc.Executable: bound executable; ``run_fitness_rule`` stages its operand.
    """
    n_policies = _positive_int(n_policies, "n_policies")
    episodes_per_policy = _positive_int(episodes_per_policy, "episodes_per_policy")
    resolved_backend, resolved_device = resolve_backend_device(backend, device)
    rows = n_policies * episodes_per_policy
    executable = evoxc.build(
        fitness_rule_graph,
        evoxc.TensorSpec((rows,), evoxc.float32, name="returns"),
        n_policies,
        episodes_per_policy,
        backend=resolved_backend,
        device=resolved_device,
    )
    return _attach(
        executable,
        graph="fitness_rule",
        rows=rows,
        n_policies=n_policies,
        episodes_per_policy=episodes_per_policy,
        device_name=_device_name(resolved_device),
    )


def run_fitness_rule(executable, returns, *, residency="numpy"):
    """Run the fitness-rule graph; ``returns`` is ``(NP * episodes,)``, result ``(NP,)``.

    Args:
        executable: the bound graph from :func:`build_fitness_rule`.
        returns: ``(NP * episodes,)`` float32, host array or device tensor.
        residency (str): ``"numpy"`` returns a host float32 array; ``"device"`` returns the
            raw device tensor with no host conversion.

    Returns:
        numpy.ndarray | evoxc.Tensor: ``(NP,)`` float32.
    """
    if _residency(residency) == "device":
        operand = _device_tensor(executable, returns, "returns", (_rows_of(executable),), dtype="floating")
        return _outputs(
            evoxc.run(executable, operand, executable.n_policies, executable.episodes_per_policy)
        )[0]
    outputs = evoxc.run(
        executable,
        _to_device_array(executable, _staged(executable, "returns", returns)),
        executable.n_policies,
        executable.episodes_per_policy,
    )
    return np.asarray(_to_numpy(_outputs(outputs)[0]), dtype=np.float32)


# --- Verification (the only device scalars read back per generation) ----------------


@evoxc.defn
def verification_graph(returns, episode_returns, fitness, n_policies, episodes, atol):
    """The three device scalars: returns tiling, fitness rule agreement, max fitness deviation."""
    tiled = evoxc.reshape(returns, (n_policies, episodes))
    same = evoxc.logical_or(
        evoxc.equal(tiled, episode_returns),
        evoxc.logical_and(evoxc.isnan(tiled), evoxc.isnan(episode_returns)),
    )
    # evoxc exposes no ``reduce_all``; a count of the satisfied entries is the same test.
    matches = evoxc.reduce_sum(evoxc.cast(same, evoxc.int32))
    returns_match = evoxc.equal(matches, n_policies * episodes)
    deviation = evoxc.abs(evoxc.subtract(fitness, _fitness_rule_ops(returns, n_policies, episodes)))
    within = evoxc.less_equal(deviation, atol)
    fitness_match = evoxc.equal(evoxc.reduce_sum(evoxc.cast(within, evoxc.int32)), n_policies)
    return [returns_match, fitness_match, evoxc.reduce_max(deviation)]


def build_verification(
    n_policies,
    *,
    episodes_per_policy=3,
    backend="numpy",
    device=None,
    atol=1e-5,
):
    """Compile the verification graph of one generation's device result.

    ``atol`` is the fitness tolerance: device residency uses ``1e-5`` because the xla
    fitness graph deviates up to 7.6e-6 from the host rule while iree is bitwise equal,
    whereas the numpy path keeps ``1e-6``.

    Args:
        n_policies (int): Population size.
        episodes_per_policy (int): Episodes per policy.
        backend (str): ``evoxc`` backend name.
        device (str | evoxc.Device | None): ``evoxc`` device.
        atol (float): Absolute fitness tolerance.

    Returns:
        evoxc.Executable: bound executable; ``run_verification`` stages its operands.
    """
    n_policies = _positive_int(n_policies, "n_policies")
    episodes_per_policy = _positive_int(episodes_per_policy, "episodes_per_policy")
    resolved_backend, resolved_device = resolve_backend_device(backend, device)
    rows = n_policies * episodes_per_policy
    specs = [
        evoxc.TensorSpec((rows,), evoxc.float32, name="returns"),
        evoxc.TensorSpec((n_policies, episodes_per_policy), evoxc.float32, name="episode_returns"),
        evoxc.TensorSpec((n_policies,), evoxc.float32, name="fitness"),
    ]
    executable = evoxc.build(
        verification_graph,
        *specs,
        n_policies,
        episodes_per_policy,
        float(atol),
        backend=resolved_backend,
        device=resolved_device,
    )
    return _attach(
        executable,
        graph="verification",
        rows=rows,
        n_policies=n_policies,
        episodes_per_policy=episodes_per_policy,
        atol=float(atol),
        device_name=_device_name(resolved_device),
    )


def run_verification(executable, *, returns, episode_returns, fitness, residency="numpy"):
    """Run the verification graph; only its three scalars cross to the host.

    Args:
        executable: the bound graph from :func:`build_verification`.
        returns: ``(rows,)`` float32, host array or device tensor.
        episode_returns: ``(NP, episodes)`` float32, host array or device tensor.
        fitness: ``(NP,)`` float32, host array or device tensor.
        residency (str): ``"numpy"`` stages host arrays; ``"device"`` requires tensors that
            are already on the executable's device and stages nothing.

    Returns:
        dict: ``returns_match`` (bool), ``fitness_match`` (bool) and
        ``fitness_max_abs_diff`` (float).
    """
    n_policies = int(executable.n_policies)
    episodes = int(executable.episodes_per_policy)
    shapes = ((int(executable.rows),), (n_policies, episodes), (n_policies,))
    operands = (returns, episode_returns, fitness)
    names = ("returns", "episode_returns", "fitness")
    if _residency(residency) == "device":
        staged = [
            _device_tensor(executable, value, name, shape, dtype="floating")
            for value, name, shape in zip(operands, names, shapes, strict=True)
        ]
    else:
        staged = [
            _staged_float_shape(executable, name, value, shape)
            for value, name, shape in zip(operands, names, shapes, strict=True)
        ]
    flat = _outputs(
        evoxc.run(
            executable,
            *staged,
            executable.n_policies,
            executable.episodes_per_policy,
            executable.atol,
        )
    )
    return {
        "returns_match": bool(_to_numpy(flat[0])),
        "fitness_match": bool(_to_numpy(flat[1])),
        "fitness_max_abs_diff": float(_to_numpy(flat[2])),
    }


# --- Tell (one generation transition) -----------------------------------------------


def _species_specs(species_leaves) -> list:
    """``TensorSpec`` list for the ``SpeciesState`` leaves, in leaf order."""
    n_fields = len(SpeciesState.__dataclass_fields__)
    leaves = tuple(species_leaves)
    if len(leaves) != n_fields:
        raise ValueError(f"species_leaves must hold the {n_fields} SpeciesState leaves, got {len(leaves)}")
    return [_spec_for(leaf, f"species_{index}") for index, leaf in enumerate(leaves)]


def build_tell(
    algo,
    config,
    pop_arrays,
    species_leaves,
    n_policies,
    *,
    backend="numpy",
    device=None,
    plugin_path=None,
):
    """Compile the TensorNEAT ``tell`` graph: payload + fitness -> next generation.

    The graph returns the flat list ``[randkey, generation, *pop, *species]``; the
    ``SpeciesState`` pytree layout is captured during the trace and stored on the
    executable, so ``run_tell`` can rebuild it without a ``state`` argument.

    Args:
        algo: a TensorNEAT ``NEAT`` algorithm instance.
        config: the tensor-free ``pipe._static_config(state)`` (must not already
            register ``pop``/``generation``/``species``/``randkey``).
        pop_arrays: the four genome payload arrays (``state.pop``), host numpy or device tensors.
        species_leaves: ``evoxc.tree_leaves(state.species)``, host numpy or device tensors.
        n_policies (int): Population size, the fitness vector length.
        backend (str): ``evoxc`` backend name.
        device (str | evoxc.Device | None): ``evoxc`` device.
        plugin_path (str | os.PathLike | None): XLA/PJRT plugin shared object; only used by
            the ``xla`` backend.  ``None`` leaves evoxc's own discovery in charge
            (``EVOXC_PJRT_PLUGIN`` or the installed ``jax_plugins/*/*.so``).

    Returns:
        evoxc.Executable: bound executable; ``run_tell`` stages its operands.
    """
    n_policies = _positive_int(n_policies, "n_policies")
    if n_policies != int(algo.pop_size):
        raise ValueError(f"n_policies must be algo.pop_size = {int(algo.pop_size)}, got {n_policies}")
    resolved_backend, resolved_device = resolve_backend_device(backend, device)
    n_leaves = len(SpeciesState.__dataclass_fields__)
    specs = [
        evoxc.TensorSpec((), evoxc.int64, name="randkey"),
        *_pop_specs(pop_arrays),
        evoxc.TensorSpec((), evoxc.float32, name="generation"),
        evoxc.TensorSpec((n_policies,), evoxc.float32, name="fitness"),
        *_species_specs(species_leaves),
    ]
    captured = []

    def kernel(randkey, node_keys, node_attrs, conn_keys, conn_attrs, generation, fitness, *species_values):
        species_tree = evoxc.tree_structure(SpeciesState(*species_values))
        if not captured:
            captured.append(species_tree)
        payload = GenomeTensors(node_keys, node_attrs, conn_keys, conn_attrs)
        species = evoxc.tree_unflatten(list(species_values), species_tree)
        current = config.register(pop=payload, generation=generation, species=species, randkey=randkey)
        nxt = algo.tell(current, fitness)
        next_species = evoxc.tree_leaves(nxt.species)
        if len(next_species) != n_leaves:
            raise ValueError(
                f"algorithm.tell changed the species pytree: {n_leaves} leaves in, {len(next_species)} out"
            )
        return [nxt.randkey, nxt.generation, *nxt.pop, *next_species]

    options = {}
    if resolved_backend == "iree" and resolved_device.kind in ("cuda", "rocm"):
        # IREE's default reduction-vector distribution rejects the tell graph on the CUDA
        # target and, on the ROCm target, leaves ``iree-compile`` spinning without
        # terminating; the same flag builds it on both.  The XLA path is unaffected.
        options["iree_compile_args"] = ["--iree-codegen-llvmgpu-use-reduction-vector-distribution=false"]
    if resolved_backend == "xla" and plugin_path is not None:
        options["plugin_path"] = str(plugin_path)
    executable = evoxc.build(kernel, *specs, backend=resolved_backend, device=resolved_device, **options)
    return _attach(
        executable,
        graph="tell",
        rows=n_policies,
        n_policies=n_policies,
        species_tree=captured[-1] if captured else None,
        species_leaf_count=n_leaves,
        pop_shapes=tuple(_shape_of(array) for array in pop_arrays),
        device_name=_device_name(resolved_device),
    )


def run_tell(executable, *, randkey, pop_arrays, generation, fitness, species_leaves, residency="numpy"):
    """Run the tell graph; returns the next ``randkey``/``generation``/``pop``/``species``.

    Args:
        executable: the bound graph from ``build_tell``.
        randkey (array-like): 0-d int64 RNG key.
        pop_arrays: the four genome payload arrays (same shapes as at build time).
        generation (array-like): 0-d float32 generation counter.
        fitness (array-like): ``(NP,)`` float32, maximized.
        species_leaves: the ``SpeciesState`` leaves (same layout as at build time).
        residency (str): ``"numpy"`` stages host arrays and materializes the results;
            ``"device"`` stages nothing (every operand must already be a tensor on the
            executable's device) and returns device tensors, so its output can be fed
            straight back into the next :func:`run_tell` call.

    Returns:
        dict: ``randkey`` (0-d int64), ``generation`` (0-d float32), ``pop`` (tuple of
        the four payload arrays) and ``species`` (rebuilt ``SpeciesState``).
    """
    if _residency(residency) == "device":
        return _run_tell_device(executable, randkey, pop_arrays, generation, fitness, species_leaves)
    arrays = tuple(_host(array) for array in pop_arrays)
    if len(arrays) != 4:
        raise ValueError(f"pop_arrays must hold the 4 GenomeTensors payloads, got {len(arrays)}")
    leaves = tuple(_host(leaf) for leaf in species_leaves)
    if len(leaves) != executable.species_leaf_count:
        raise ValueError(
            f"species_leaves must hold {executable.species_leaf_count} leaves, got {len(leaves)}"
        )
    for array, shape in zip(arrays, executable.pop_shapes):
        if array.shape != shape:
            raise ValueError(f"pop_arrays leaf must keep the built shape {shape}, got {array.shape}")
    n_policies = int(executable.n_policies)
    key = _host(randkey).astype(np.int64, copy=False)
    if key.shape != ():
        raise ValueError(f"randkey must be a 0-d int64 key, got shape {key.shape}")
    gen = _host(generation).astype(np.float32, copy=False)
    if gen.shape != ():
        raise ValueError(f"generation must be a 0-d float32 counter, got shape {gen.shape}")
    fit = _host(fitness).astype(np.float32, copy=False)
    if fit.shape != (n_policies,):
        raise ValueError(f"fitness must have shape ({n_policies},), got {fit.shape}")
    outputs = evoxc.run(
        executable,
        _to_device_array(executable, key),
        *[_to_device_array(executable, np.ascontiguousarray(array)) for array in arrays],
        _to_device_array(executable, gen),
        _to_device_array(executable, np.ascontiguousarray(fit)),
        *[_to_device_array(executable, _contiguous(leaf)) for leaf in leaves],
    )
    n_leaves = int(executable.species_leaf_count)
    flat = _outputs(outputs)
    next_pop = tuple(_to_numpy(flat[2 + index]) for index in range(4))
    next_species = [_to_numpy(flat[6 + index]) for index in range(n_leaves)]
    return {
        "randkey": _to_numpy(flat[0]),
        "generation": _to_numpy(flat[1]),
        "pop": next_pop,
        "species": evoxc.tree_unflatten(next_species, executable.species_tree),
    }


def _run_tell_device(executable, randkey, pop_arrays, generation, fitness, species_leaves):
    """Device-resident :func:`run_tell`: validate from metadata, stage nothing, return device tensors."""
    arrays = tuple(pop_arrays)
    if len(arrays) != len(POP_OPERANDS):
        raise ValueError(f"pop_arrays must hold the {len(POP_OPERANDS)} GenomeTensors payloads, got {len(arrays)}")
    for array, name, shape in zip(arrays, POP_OPERANDS, executable.pop_shapes, strict=True):
        _device_tensor(executable, array, name, shape)
    leaves = tuple(species_leaves)
    if len(leaves) != int(executable.species_leaf_count):
        raise ValueError(
            f"species_leaves must hold {executable.species_leaf_count} leaves, got {len(leaves)}"
        )
    for index, leaf in enumerate(leaves):
        _device_tensor(executable, leaf, f"species_{index}")
    n_policies = int(executable.n_policies)
    key = _device_tensor(executable, randkey, "randkey", (), dtype="integer")
    gen = _device_tensor(executable, generation, "generation", (), dtype="floating")
    fit = _device_tensor(executable, fitness, "fitness", (n_policies,), dtype="floating")
    flat = _outputs(evoxc.run(executable, key, *arrays, gen, fit, *leaves))
    n_leaves = int(executable.species_leaf_count)
    return {
        "randkey": flat[0],
        "generation": flat[1],
        "pop": tuple(flat[2 + index] for index in range(len(POP_OPERANDS))),
        "species": evoxc.tree_unflatten(
            [flat[2 + len(POP_OPERANDS) + index] for index in range(n_leaves)], executable.species_tree
        ),
    }


# --- Executable cache ---------------------------------------------------------------


class ExecutableCache:
    """Shape-keyed cache of bound executables, so a generation only builds once.

    Keys are hashable tuples of the static dimensions that specialize a graph --
    graph kind, row count, episode count, resolved backend, resolved device name --
    e.g. ``("fitness", rows, episodes, backend, device_name)``.  Same key -> same
    executable; the next generation of the same shape hits the cache.  A hit/miss
    debug line is logged per call.
    """

    def __init__(self):
        self._executables = {}
        self.hits = 0
        self.misses = 0

    def get(self, key, builder):
        """Return the executable for ``key``, building it with ``builder()`` on a miss.

        Args:
            key: hashable tuple of static dimensions.
            builder: zero-argument callable returning a bound executable.

        Returns:
            evoxc.Executable: the cached (or freshly built) executable.
        """
        key = tuple(key)
        cached = self._executables.get(key)
        if cached is not None:
            self.hits += 1
            logger.debug("executable cache hit: %s", key)
            return cached
        self.misses += 1
        logger.debug("executable cache miss: %s", key)
        executable = builder()
        self._executables[key] = executable
        return executable

    def clear(self):
        """Drop every cached executable and reset the hit/miss counters."""
        self._executables.clear()
        self.hits = 0
        self.misses = 0
