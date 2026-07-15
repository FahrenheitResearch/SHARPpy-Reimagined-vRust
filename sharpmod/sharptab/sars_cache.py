"""Cached adapters for the vendored SHARPpy SARS match functions.

The upstream functions parse the same two immutable text databases on every
call -- four parses for every convective profile because right- and left-moving
matches are evaluated separately.  These adapters execute the exact vendored
function bodies while substituting only a cached ``numpy.loadtxt`` boundary.
"""

from __future__ import annotations

from functools import lru_cache
import os
from types import FunctionType

import numpy as np
from sharppy.databases import sars as _upstream


_ORIGINAL_LOADTXT = np.loadtxt


@lru_cache(maxsize=4)
def _cached_loadtxt(path, args, kwargs):
    """Read one immutable SARS table once per process."""
    return _ORIGINAL_LOADTXT(path, *args, **dict(kwargs))


class _CachedNumpy:
    """Forward NumPy operations while caching hashable ``loadtxt`` calls."""

    def __getattr__(self, name):
        return getattr(np, name)

    def loadtxt(self, path, *args, **kwargs):
        key = (os.fspath(path), tuple(args), tuple(sorted(kwargs.items())))
        try:
            hash(key)
        except TypeError:
            return _ORIGINAL_LOADTXT(path, *args, **kwargs)
        return _cached_loadtxt(*key)


_CACHED_NUMPY = _CachedNumpy()


def _with_cached_database(function):
    """Clone a vendored function with only its ``np`` global replaced."""
    globals_ = dict(function.__globals__)
    globals_["np"] = _CACHED_NUMPY
    adapted = FunctionType(
        function.__code__, globals_, function.__name__, function.__defaults__,
        function.__closure__)
    adapted.__kwdefaults__ = function.__kwdefaults__
    adapted.__doc__ = function.__doc__
    return adapted


hail = _with_cached_database(_upstream.hail)
supercell = _with_cached_database(_upstream.supercell)


def clear_cache():
    """Clear parsed tables for deterministic tests."""
    _cached_loadtxt.cache_clear()

