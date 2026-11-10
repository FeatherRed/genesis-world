"""The device-residency plumbing of the neuroevolution layer: placement, key, device-first init.

Device residency means one thing everywhere in this directory: the four
``GenomeTensors`` payload leaves, the ``SpeciesState`` leaves, the RNG key and the
generation counter are ``evoxc`` tensors that live on one device, and no host
roundtrip happens between two generations.
This module owns the reusable pieces of that statement:

* the placement predicates and assertions (:func:`is_device_tensor`,
  :func:`same_device`, :func:`device_placement`, :func:`assert_device_resident`,
  :func:`assert_host_value`, :func:`placement_map`,
  :func:`assert_payload_resident`), which read tensor metadata only;
* :func:`device_key`, the device-resident ``evoxc.random.key`` seed of a run;
* :func:`host_formulation_statics` and :func:`device_setup`, the device-first
  initialization: a generation-0 state built by ``algo.setup`` on the target
  device instead of ``Pipeline.setup()`` plus ``_place_state``'s per-leaf
  transfer;
* :func:`device_payload` / :func:`device_species_leaves`, the leaves a trainer
  hands to the tell graph without materializing them.

Nothing here imports ``torch``, ``genesis`` or ``tensordict``; the imports are
``numpy``, ``evoxc``, ``TensorNEAT`` and this directory's own modules.

Why the host formulation triple
-------------------------------
``device_setup`` registers ``("single", "merge", "host")`` -- the triple a numpy
``Pipeline`` derives -- rather than the device-derived GPU triple: the tell graph
must compile at the population sizes the pilots use, and the GPU triple
``("tiled", "index_free", "accelerator")`` does not compile on ``xla@rocm:0`` at
pop 8/16/32 while the host triple does at every tested pop.
The triple is a trace-time static, so it is registered before ``algo.setup`` and
travels into the static config the tell graph is built from.

One xla caveat
--------------
:func:`device_key` places the key through ``Tensor.to``, whose xla provider only
sees the options-less plugin discovery chain (``EVOXC_PJRT_PLUGIN`` or an installed
``jax_plugins/*/*.so``): the ``plugin_path`` compile option reaches ``evoxc.build``
but never the placement provider, so device-first initialization on ``xla@rocm:0``
needs the environment variable (or a ``sys.path`` plugin) even when the tell graph
is built with ``plugin_path=``.
"""

from __future__ import annotations

import numpy as np

import evoxc

from tensorneat import Pipeline
from tensorneat.common import State

from env_base import RESIDENCIES
from evox_runtime import POP_OPERANDS, resolve_backend_device

__all__ = [
    "RESIDENCIES",
    "assert_device_resident",
    "assert_host_value",
    "assert_payload_resident",
    "device_key",
    "device_payload",
    "device_placement",
    "device_setup",
    "device_species_leaves",
    "host_formulation_statics",
    "is_device_tensor",
    "placement_map",
    "same_device",
]


# --- Placement predicates -----------------------------------------------------------


def is_device_tensor(value) -> bool:
    """True when ``value`` is an ``evoxc`` tensor (has ``.shape``/``.dtype``/``.device``), never materialized."""
    if isinstance(value, np.ndarray):
        return False
    return all(hasattr(value, attribute) for attribute in ("shape", "dtype", "device"))


def _placement(device) -> str:
    """``"kind:index"`` of an ``evoxc`` device (or the plain ``str`` of anything else)."""
    kind = getattr(device, "kind", None)
    if kind is None:
        return str(device)
    return f"{kind}:{getattr(device, 'index', 0)}"


def device_placement(value) -> str:
    """``"rocm:0"``-style placement of a tensor, or ``"host"`` when it is not device-resident."""
    if not is_device_tensor(value):
        return "host"
    return _placement(value.device)


def same_device(value, device) -> bool:
    """True when ``value`` is a device tensor placed on exactly ``device``."""
    return is_device_tensor(value) and device_placement(value) == _placement(device)


def assert_device_resident(value, device, name="value"):
    """Raise ``TypeError`` unless ``value`` is a tensor placed on exactly ``device``.

    Args:
        value: The value to check; nothing is materialized.
        device (evoxc.Device): The expected placement.
        name (str): Name used in the failure message.

    Returns:
        The value, unchanged.
    """
    target = _placement(device)
    if not is_device_tensor(value):
        raise TypeError(
            f"{name} must be a device-resident tensor on {target}, got {type(value).__name__}; "
            "device residency never stages a host value"
        )
    actual = device_placement(value)
    if actual != target:
        payload = type(getattr(value, "data", None)).__name__
        raise TypeError(
            f"{name} is on {actual} carrying {payload}, but the device residency of this run is {target}"
        )
    return value


def assert_host_value(value, name="value"):
    """Raise ``TypeError`` when ``value`` is a device tensor, i.e. when a host value was expected.

    Args:
        value: The value to check.
        name (str): Name used in the failure message.

    Returns:
        The value, unchanged.
    """
    if is_device_tensor(value):
        raise TypeError(
            f"{name} must be a host value, got a tensor on {device_placement(value)}; move it with "
            "`.to(evoxc.Device('cpu', 0))` and materialize it only where the contract allows it"
        )
    return value


def placement_map(values, names=None) -> dict:
    """Map each value's name to its placement, for logging and placement assertions.

    Args:
        values: A mapping of name -> value, or a sequence of values.
        names (sequence[str] | None): Names of a sequence's entries; defaults to ``str(index)``.

    Returns:
        dict: ``{name: "rocm:0" | "cpu:0" | "host"}``.
    """
    if isinstance(values, dict):
        return {str(name): device_placement(value) for name, value in values.items()}
    entries = tuple(values)
    labels = [str(index) for index in range(len(entries))] if names is None else [str(name) for name in names]
    if len(labels) != len(entries):
        raise ValueError(f"names must name every value: {len(labels)} names for {len(entries)} values")
    return {label: device_placement(value) for label, value in zip(labels, entries, strict=True)}


# --- Device-resident RNG key --------------------------------------------------------


def device_key(seed, backend, device):
    """The device-resident ``evoxc.random.key(seed)`` of a run: 0-d int64 on ``device``.

    The transfer provider behind ``Tensor.to`` belongs to the backend that was looked up
    last, so the backend is re-selected before the key is placed; in a mixed-backend
    process an IREE run must not inherit the XLA adapter's provider (or the reverse).

    Args:
        seed (int): The run's seed.
        backend (str): ``evoxc`` backend name.
        device (evoxc.Device | str | None): Target device.

    Returns:
        evoxc.Tensor: A 0-d int64 key on ``device``, equal in value/dtype to ``evoxc.random.key(seed)``.
    """
    resolved_backend, resolved_device = resolve_backend_device(backend, device)
    evoxc.backends.get(resolved_backend)
    return evoxc.random.key(int(seed)).to(resolved_device)


# --- Device-first initialization ----------------------------------------------------


def host_formulation_statics(algo, problem):
    """The host kernel-formulation triple ``(distance_centre, distance_formulation, kernel_mode)``.

    A numpy ``Pipeline`` is constructed for this only -- never ``setup()`` -- and its
    public attributes are read back, because ``Pipeline._static_config`` needs the
    triple on the state and the device-derived GPU triple does not compile the tell
    graph at the pilots' population sizes.

    Args:
        algo: A TensorNEAT algorithm instance.
        problem: The pipeline's problem (only its declared widths are read).

    Returns:
        tuple: ``("single", "merge", "host")`` for a ``None`` (host) device.
    """
    pipeline = Pipeline(algo, problem, backend="numpy", is_detailed=False)
    return pipeline.distance_centre, pipeline.distance_formulation, pipeline.kernel_mode


def device_setup(algo, problem, *, seed, backend, device, compile_options=None):
    """Build a generation-0 state that is device-resident from the start.

    The order is ``randkey``, then the compile placement and the formulation statics,
    then ``algo.setup`` (TensorNEAT's own initialization graph, which compiles and runs
    on the registered backend/device), then ``problem.setup``, then the tensor-free
    static config the compiled graphs are traced with.
    The CPU equivalent of this path is bitwise identical to ``Pipeline.setup()``.

    Args:
        algo: A TensorNEAT algorithm instance.
        problem: The pipeline's problem; ``problem.setup`` is the inherited identity.
        seed (int): The evolution stream's seed (``seeding.evolution_seed``).
        backend (str): ``evoxc`` backend name.
        device (evoxc.Device | str | None): Target device.
        compile_options (Mapping | None): Backend compile/load options registered on the state.

    Returns:
        tuple: ``(config, state)`` -- the tensor-free static config and the device-resident state.
    """
    resolved_backend, resolved_device = resolve_backend_device(backend, device)
    centre, formulation, mode = host_formulation_statics(algo, problem)
    state = State()
    state = state.register(randkey=device_key(seed, resolved_backend, resolved_device))
    state = state.register(
        backend=resolved_backend,
        device=resolved_device,
        compile_options=None if compile_options is None else dict(compile_options),
        distance_centre=centre,
        distance_formulation=formulation,
        kernel_mode=mode,
    )
    state = algo.setup(state)
    state = problem.setup(state)
    pipeline = Pipeline(
        algo,
        problem,
        seed=int(seed),
        backend=resolved_backend,
        device=resolved_device,
        is_detailed=False,
    )
    extractor = getattr(pipeline, "_static_config", None)
    if not callable(extractor):
        raise RuntimeError(
            "TensorNEAT's private Pipeline._static_config is missing; device-first initialization needs "
            "it to trace the compiled graphs (pinned TensorNEAT version changed?)"
        )
    return extractor(state), state


def device_payload(state):
    """The four genome payload leaves of a state, in ``GenomeTensors`` order, unmaterialized.

    Args:
        state: The state returned by :func:`device_setup` (or by ``run_tell``).

    Returns:
        tuple: ``(node_keys, node_attrs, conn_keys, conn_attrs)`` as device tensors.
    """
    return tuple(state.pop)


def device_species_leaves(state):
    """The ``SpeciesState`` leaves of a state, in ``evoxc.tree_leaves`` order, unmaterialized.

    Args:
        state: The state returned by :func:`device_setup` (or by ``run_tell``).

    Returns:
        tuple: The species leaves as device tensors.
    """
    return tuple(evoxc.tree_leaves(state.species))


def assert_payload_resident(pop_arrays, species_leaves, randkey, generation, device):
    """Raise ``TypeError`` unless every runtime leaf of a generation lives on ``device``.

    Checks the four payload leaves, every species leaf, the RNG key and the generation
    counter by metadata only.

    Args:
        pop_arrays: The four genome payload leaves.
        species_leaves: The ``SpeciesState`` leaves.
        randkey: The 0-d int64 RNG key.
        generation: The 0-d float32 generation counter.
        device (evoxc.Device): The expected placement.

    Returns:
        None: Every leaf is device-resident.
    """
    payload = tuple(pop_arrays)
    if len(payload) != len(POP_OPERANDS):
        raise ValueError(f"pop_arrays must hold the {len(POP_OPERANDS)} GenomeTensors payloads, got {len(payload)}")
    for value, name in zip(payload, POP_OPERANDS, strict=True):
        assert_device_resident(value, device, name)
    for index, value in enumerate(species_leaves):
        assert_device_resident(value, device, f"species_{index}")
    assert_device_resident(randkey, device, "randkey")
    assert_device_resident(generation, device, "generation")
