"""Task-agnostic evolution-side ``evoxc`` graphs for the TensorNEAT trainers.

Two graphs live here, plus the shape-keyed executable cache the trainers use so a
generation only ever builds once:

``fitness_graph``
    Episode returns ``(n_policies * episodes_per_policy,)`` float32 -> per-policy
    fitness ``(n_policies,)`` float32, the mean over each policy's episodes.

``tell_graph``
    One TensorNEAT ``tell`` transition: ``randkey``, the four genome payload
    arrays, ``generation``, ``fitness`` and the ``SpeciesState`` leaves in, the
    flat list ``[randkey, generation, *pop, *species]`` out.

Neither graph knows a task: no observation width, action width, reward or
simulator constant appears here, so the Go2 walking (45 -> 12) and the drone
hover (17 -> 4) trainers compile the very same code.

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

from evox_runtime import (
    _attach,
    _contiguous,
    _device_name,
    _host,
    _outputs,
    _pop_specs,
    _positive_int,
    _spec_for,
    _staged,
    _to_device_array,
    _to_numpy,
    resolve_backend_device,
)

logger = logging.getLogger(__name__)


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


def run_fitness(executable, returns):
    """Run the fitness graph; ``returns`` is ``(NP * episodes,)``, result ``(NP,)``."""
    outputs = evoxc.run(
        executable,
        _to_device_array(executable, _staged(executable, "returns", returns)),
        executable.n_policies,
        executable.episodes_per_policy,
    )
    return np.asarray(_to_numpy(_outputs(outputs)[0]), dtype=np.float32)


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
        pop_arrays: the four genome payload arrays (``state.pop`` as numpy).
        species_leaves: ``evoxc.tree_leaves(state.species)`` as numpy.
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
    if resolved_backend == "iree" and resolved_device.kind == "cuda":
        # IREE 3.13's default CUDA reduction distribution rejects the tell graph.
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
        pop_shapes=tuple(_host(array).shape for array in pop_arrays),
        device_name=_device_name(resolved_device),
    )


def run_tell(executable, *, randkey, pop_arrays, generation, fitness, species_leaves):
    """Run the tell graph; returns the next ``randkey``/``generation``/``pop``/``species``.

    Args:
        executable: the bound graph from ``build_tell``.
        randkey (array-like): 0-d int64 RNG key.
        pop_arrays: the four genome payload arrays (same shapes as at build time).
        generation (array-like): 0-d float32 generation counter.
        fitness (array-like): ``(NP,)`` float32, maximized.
        species_leaves: the ``SpeciesState`` leaves (same layout as at build time).

    Returns:
        dict: ``randkey`` (0-d int64), ``generation`` (0-d float32), ``pop`` (tuple of
        the four payload arrays) and ``species`` (rebuilt ``SpeciesState``).
    """
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
