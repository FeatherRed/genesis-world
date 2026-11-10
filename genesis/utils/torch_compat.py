"""Optional-PyTorch compatibility layer, the single import site for PyTorch in Genesis.

Genesis must run where PyTorch is not installed: the simulation core and the device-buffer bridge only need
Quadrants and NumPy. Modules therefore import PyTorch from here instead of importing it directly::

    from genesis.utils.torch_compat import HAS_TORCH, no_grad, require_torch, torch

When PyTorch is available, this module re-exports the real module and nothing else changes. Otherwise ``torch`` is a
module-like placeholder: every attribute access yields a sentinel *type* (:class:`_Unavailable`), so annotations such
as ``torch.Tensor``, unions such as ``torch.Tensor | None`` and subscriptions such as ``torch.Tensor[(3,)]`` all keep
evaluating at import time, ``isinstance(value, torch.Tensor)`` remains a legal call returning False, and calling a
sentinel such as ``torch.zeros(3)`` raises a Genesis error naming the missing feature instead of a bare TypeError.
"""

from __future__ import annotations

import contextlib
from typing import NoReturn

__all__ = ["HAS_TORCH", "torch", "require_torch", "no_grad", "is_torch_tensor"]

try:
    import torch
except ImportError:
    HAS_TORCH = False
else:
    HAS_TORCH = True


def _missing_torch_message(feature: str) -> str:
    return (
        f"'{feature}' requires PyTorch, which is not installed. Feature: {feature}. "
        "Install PyTorch following the instructions at https://pytorch.org/get-started/locally/"
    )


def _raise_genesis_error(message: str, cause: BaseException | None = None) -> NoReturn:
    """Raise a Genesis error, falling back to ImportError while the package entry point still requires PyTorch.

    Importing 'genesis' is deferred to call time, both because this module is imported while 'genesis' itself is
    loading and because 'genesis/__init__.py' imports PyTorch eagerly for now. In the intermediate state where
    PyTorch is missing, no GenesisException can be constructed, so the actionable message is raised as an
    ImportError instead; it becomes a GenesisException as soon as the package entry point is torch-free too.
    """
    try:
        import genesis as gs
    except ImportError:
        raise ImportError(message) from None
    if cause is None:
        gs.raise_exception(message)
    gs.raise_exception_from(message, cause)


class _UnavailableMeta(type):
    """Metaclass giving the sentinel of a missing PyTorch symbol its class-level behaviour."""

    def __call__(cls, *args, **kwargs):
        _raise_genesis_error(_missing_torch_message(cls._feature))

    def __getattr__(cls, name):
        # The protocols that type annotations and the standard library probe are dunders, and answering those with a
        # sentinel instead of failing would break pretty-printing and pickling of any class annotated with one.
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(f"'{cls._feature}.{name}' is unavailable because PyTorch is not installed.")
        return _unavailable_symbol(f"{cls._feature}.{name}")

    def __getitem__(cls, key):
        # Covers 'torch.Tensor[(3,)]' as well as a bare 'torch.Tensor[...]' subscription.
        return cls

    def __repr__(cls):
        return f"<unavailable PyTorch symbol '{cls._feature}'>"


class _Unavailable(metaclass=_UnavailableMeta):
    """Stand-in type for every PyTorch symbol, so annotations and 'isinstance' checks keep working."""

    _feature = "torch"


_UNAVAILABLE_SYMBOLS: dict[str, type] = {}


def _unavailable_symbol(feature: str) -> type:
    """Sentinel type for one PyTorch symbol, created once per dotted path so that identity is stable."""
    symbol = _UNAVAILABLE_SYMBOLS.get(feature)
    if symbol is None:
        name = feature.rsplit(".", 1)[-1]
        symbol = _UnavailableMeta(
            name if name.isidentifier() else "Unavailable",
            (_Unavailable,),
            {"_feature": feature, "__module__": __name__},
        )
        _UNAVAILABLE_SYMBOLS[feature] = symbol
    return symbol


class _TorchPlaceholder:
    """Module-like stand-in for 'torch' whose attribute access yields missing-symbol sentinels."""

    __version__ = "0.0.0"

    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(f"'torch.{name}' is unavailable because PyTorch is not installed.")
        return _unavailable_symbol(f"torch.{name}")

    def __repr__(self):
        return "<torch placeholder: PyTorch is not installed>"


if not HAS_TORCH:
    torch = _TorchPlaceholder()


def require_torch(feature: str) -> None:
    """Raise a Genesis error naming 'feature' if PyTorch is not installed, and do nothing otherwise."""
    if not HAS_TORCH:
        _raise_genesis_error(_missing_torch_message(feature))


class _NullGradContext(contextlib.ContextDecorator):
    """No-op stand-in for 'torch.no_grad()', usable both as a context manager and as a decorator."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


def no_grad():
    """Disable gradient tracking, as 'torch.no_grad()' does, and do nothing when PyTorch is not installed."""
    if HAS_TORCH:
        return torch.no_grad()
    return _NullGradContext()


def is_torch_tensor(value) -> bool:
    """Whether 'value' is a PyTorch tensor, without raising when PyTorch is not installed."""
    return HAS_TORCH and isinstance(value, torch.Tensor)
