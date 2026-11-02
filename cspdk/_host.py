"""Model-host selector: pick which S-parameter engine the cspdk models run on.

cspdk's model library is written against one small, structural (duck-typed) protocol so
the *same* model code can drive several engines instead of a hard ``sax`` dependency. An
engine exposes a *host* module (``sax.host`` for the default host, or another engine's)
carrying:

``xp``
    the array module the models are written against (``jax.numpy`` on the default host,
    ``numpy`` elsewhere);
``sm``
    the engine's model library;
``solve(connections, ports, instances)``
    a nested circuit solve;
``port_namer``
    the engine's port-naming strategy (carried per host, not as a module global);
``differentiable``
    whether the engine traces differentiably;
``interpolate``
    the engine's table reader;
``jit`` / ``ensure_compile_time_eval``
    the engine's tracing decorator / compile-time guard (a passthrough / no-op where the
    engine does not trace), so a model module never imports ``jax``;
``reciprocal``
    the engine's S-matrix reciprocity helper, so a model module never imports ``sax``;
``set_port_naming_strategy(strategy)``
    apply a port-naming strategy to the engine's *own* models (however that host
    carries it), so a model module never imports ``sax``;
``SDict`` / ``SCoo`` / ``SType`` / ``FloatArrayLike`` / ``FloatArray`` / ``ArrayLike``
    the type aliases the models annotate with.

This module is a **selector only**. It resolves the current host on demand, never at
import time, so:

* ``import cspdk`` (and importing a model module) does not import ``sax`` or ``jax``
  eagerly -- a process without jax still imports cspdk; and
* :func:`set_host` may be called before the first model call.

Resolution order (first hit wins):

1. an explicit host set with :func:`set_host`;
2. a host set for the current :func:`host_scope` (a ``contextvars`` scope);
3. the ``GFP_MODEL_HOST`` environment variable (read **once**, at import);
4. the lazy default: the host shipped by ``sax``.

``sm``/``xp``/``solve``/``port_namer``/``interpolate``/``jit``/``ensure_compile_time_eval``/
``reciprocal``/``set_port_naming_strategy`` and the type aliases are thin proxy objects that
forward attribute access to the resolved host, so model modules keep their existing call
sites (``sm.straight(...)``, ``jnp.sin(...)``, ``reciprocal(...)``) and nothing binds at
import time. The resolved host is cached, so the forwarding cost is a single attribute hop
per access, not per array operation.
"""

from __future__ import annotations

import contextlib
import contextvars
import importlib
import os
from typing import Any

__all__ = [
    "ArrayLike",
    "FloatArray",
    "FloatArrayLike",
    "SCoo",
    "SDict",
    "SType",
    "differentiable",
    "ensure_compile_time_eval",
    "get_host",
    "host_scope",
    "interpolate",
    "jit",
    "port_namer",
    "reciprocal",
    "set_host",
    "set_port_naming_strategy",
    "sm",
    "solve",
    "xp",
]

ENV_VAR = "GFP_MODEL_HOST"

# The env var is read once, at import, and used as the *initial* value; later reads of the
# environment do not change an already-resolved host. Its value is a dotted module path.
_ENV_INITIAL: str | None = os.environ.get(ENV_VAR) or None

# The lazy default: the host module shipped by sax. cspdk knows no engine names, so a host
# is supplied either as an object (``set_host``) or as a dotted module path (the env var,
# ``set_host`` or ``host_scope``).
_DEFAULT = "sax.host"

_explicit: Any = None
_scope: contextvars.ContextVar[Any] = contextvars.ContextVar(
    "cspdk_model_host", default=None
)
_resolved: Any = None


def _load(host: Any) -> Any:
    """Turn a host reference (object, or a dotted module path) into a host module."""
    if host is None:
        return None
    if isinstance(host, str):
        return importlib.import_module(host)  # a dotted module path
    return host  # already a host module or object


def _resolve() -> Any:
    """Resolve the current host, importing it lazily."""
    for candidate in (_explicit, _scope.get(), _ENV_INITIAL):
        host = _load(candidate)
        if host is not None:
            return host
    return _load(_DEFAULT)


def get_host() -> Any:
    """Return the host module currently in effect (importing it on first use)."""
    global _resolved
    if _resolved is None:
        _resolved = _resolve()
    return _resolved


def _invalidate() -> None:
    global _resolved
    _resolved = None


def set_host(host: Any) -> Any:
    """Set the process-wide model host.

    Args:
        host: a host module / object, or a dotted module path (``"my.engine.host"``).
            ``None`` clears the override, so resolution falls back to the scope, then
            the env var, then the default host.

    Returns:
        The host argument, for chaining. Resolution is lazy: nothing is imported here.
    """
    global _explicit
    _explicit = host
    _invalidate()
    return host


@contextlib.contextmanager
def host_scope(host: Any):
    """Temporarily set the model host for the duration of a ``with`` block."""
    token = _scope.set(host)
    _invalidate()
    try:
        yield host
    finally:
        _scope.reset(token)
        _invalidate()


class _Proxy:
    """Forwards attribute access to the current host's ``member``.

    Nothing binds at import time: the host is resolved on each access (through
    :func:`get_host`, which caches the resolved module), so ``sm.straight`` becomes
    ``get_host().sm.straight`` and the cost is one attribute hop per access.
    """

    __slots__ = ("_member",)

    def __init__(self, member: str) -> None:
        self._member = member

    def _target(self) -> Any:
        return getattr(get_host(), self._member)

    def __getattr__(self, attr: str) -> Any:
        return getattr(self._target(), attr)

    def __getitem__(self, key: Any) -> Any:
        return self._target()[key]

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._target()(*args, **kwargs)

    def __repr__(self) -> str:
        try:
            return f"cspdk._host.{self._member} -> {self._target()!r}"
        except Exception:  # pragma: no cover - only for nicer debugging
            return f"cspdk._host.{self._member}"


sm = _Proxy("sm")
xp = _Proxy("xp")
solve = _Proxy("solve")
port_namer = _Proxy("port_namer")
interpolate = _Proxy("interpolate")

# The model-writing surface: tracing, reciprocity, port-naming on the engine's own models,
# and the type aliases -- all forwarded, so a model module never imports jax or sax.
jit = _Proxy("jit")
ensure_compile_time_eval = _Proxy("ensure_compile_time_eval")
reciprocal = _Proxy("reciprocal")
set_port_naming_strategy = _Proxy("set_port_naming_strategy")
SDict = _Proxy("SDict")
SCoo = _Proxy("SCoo")
SType = _Proxy("SType")
FloatArrayLike = _Proxy("FloatArrayLike")
FloatArray = _Proxy("FloatArray")
ArrayLike = _Proxy("ArrayLike")


def differentiable() -> bool:
    """Return whether the current host traces differentiably."""
    return bool(get_host().differentiable)
