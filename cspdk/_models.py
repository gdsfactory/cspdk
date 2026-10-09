"""Private helpers shared by the flavour SAX models.

Path lengths are written with ``jax.numpy`` and have no Python branching on
their arguments, so circulax and ``jax.jit`` can trace every setting.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable, Mapping
from functools import partial, wraps
from typing import Any

import jax.numpy as jnp
import numpy as np
import sax

# 32-point Gauss-Legendre rule for the Fresnel integrals of the Euler bend.
_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(32)


def _optical_model(model: Callable, inputs: int, outputs: int) -> Callable:
    """Normalize SAX ports without changing its process-wide naming strategy.

    Translate input/output and zero-based optical keys, preserving one-based
    optical keys. Inspect the returned keys because jitted models may retain
    a naming convention cached before the current strategy was selected.
    """
    port_map = {f"in{i}": f"o{i + 1}" for i in range(inputs)}
    port_map.update({f"out{i}": f"o{inputs + outputs - i}" for i in range(outputs)})

    @wraps(model)
    def optical(*args, **kwargs):
        result = model(*args, **kwargs)
        mapping = port_map
        if any("o0" in pair for pair in result):
            mapping = {
                **port_map,
                **{f"o{i}": f"o{i + 1}" for i in range(inputs + outputs)},
            }
        return {
            (mapping.get(p, p), mapping.get(q, q)): value
            for (p, q), value in result.items()
        }

    return optical


def _euler_length(radius: Any, angle: Any, p: Any) -> Any:
    """Centre-line length of ``gf.path.euler(radius, angle, p, use_eff=True)``.

    This is the PDK Euler bend (``with_arc_floorplan=True``): the unit
    clothoid/arc/clothoid curve has length ``s0`` and is scaled so its end
    points match an arc of ``radius``. The Fresnel integrals use a 32-point
    Gauss-Legendre rule (agrees with gdsfactory to ~1e-8 um for p in
    [0.05, 1], angles up to 180 degrees and radii up to 300 um).

    Args:
        radius: effective bend radius in um.
        angle: bend angle in degrees.
        p: fraction of the bend that is an Euler curve; p -> 0 is a circular arc.
    """
    alpha = jnp.deg2rad(jnp.abs(angle))
    p = jnp.clip(p, 1e-12, 1.0)
    sp = jnp.sqrt(p * alpha)
    rp = 1 / sp
    # Clothoid end point: sqrt(2) * int_0^u (cos t^2, sin t^2) dt, u = sp/sqrt(2).
    u = sp / np.sqrt(2)
    t = u * (_GL_NODES + 1) / 2
    w = u * _GL_WEIGHTS / 2
    xp = np.sqrt(2) * jnp.sum(w * jnp.cos(t**2))
    yp = np.sqrt(2) * jnp.sum(w * jnp.sin(t**2))
    # Midpoint of the symmetric curve (end of the half arc at alpha/2).
    x1 = rp * (jnp.sin(alpha / 2) - jnp.sin(p * alpha / 2)) + xp
    y1 = rp * (jnp.cos(p * alpha / 2) - jnp.cos(alpha / 2)) + yp
    r_eff = (x1 * jnp.cos(alpha / 2) + y1 * jnp.sin(alpha / 2)) / jnp.sin(alpha / 2)
    s0 = 2 * sp + rp * alpha * (1 - p)
    return s0 * radius / r_eff


def _bend_s_length(dx: Any, dy: Any, npoints: int = 99) -> Any:
    """Centre-line length of ``gf.components.bend_s(size=(dx, dy))``.

    gdsfactory draws the cubic Bezier with control points (0, 0), (dx/2, 0),
    (dx/2, dy), (dx, dy) as an ``npoints`` polyline (99 by default) and
    reports that polyline's length, which this reproduces exactly.

    Args:
        dx: S-bend length in um.
        dy: S-bend height in um.
        npoints: number of polyline points (not traced).
    """
    t = np.linspace(0, 1, npoints)
    x = (3 * (1 - t) ** 2 * t + 3 * (1 - t) * t**2) * dx / 2 + t**3 * dx
    y = (3 * (1 - t) * t**2 + t**3) * dy
    return jnp.sum(jnp.hypot(jnp.diff(x), jnp.diff(y)))


def _sdict_models(
    namespace: Mapping[str, Any], exclude: Iterable[str] = ()
) -> dict[str, Callable]:
    """Return the public callables of a models module annotated ``-> SDict``.

    Args:
        namespace: the module's ``globals()``.
        exclude: public names to leave out.
    """
    exclude = set(exclude)
    models = {}
    for name, func in list(namespace.items()):
        if name.startswith("_") or name in exclude or not callable(func):
            continue
        _func = func
        while isinstance(_func, partial):
            _func = _func.func
        try:
            sig = inspect.signature(_func)
        except (ValueError, TypeError):
            continue
        if (
            sig.return_annotation == sax.SDict
            or str(sig.return_annotation).lower().split(".")[-1] == "sdict"
        ):
            models[name] = func
    return models
