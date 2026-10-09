"""The shared SAX model helpers (cspdk._models) against gdsfactory."""

from __future__ import annotations

from functools import partial

import gdsfactory as gf
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import sax
import sax.models as sm
from gdsfactory.components.bends.bend_s import bezier_curve

from cspdk._models import _bend_s_length, _euler_length, _optical_model, _sdict_models
from cspdk.si340 import PDK


@pytest.fixture(autouse=True)
def activate_pdk():
    """Draw the reference S-bends with a cspdk cross-section."""
    PDK.activate()


def _polyline_length(points) -> float:
    return float(np.sum(np.hypot(*np.diff(np.asarray(points), axis=0).T)))


@pytest.mark.parametrize("radius", [10.0, 300.0])
@pytest.mark.parametrize("p", np.linspace(0.05, 1.0, 20).round(2).tolist())
def test_euler_length_matches_gdsfactory(radius, p):
    """Exact length of gf.path.euler(use_eff=True) for angles 15..180 degrees."""
    for angle in range(15, 181, 15):
        path = gf.path.euler(
            radius=radius, angle=angle, p=p, use_eff=True, npoints=100_000
        )
        expected = _polyline_length(path.points)
        assert float(_euler_length(radius, angle, p)) == pytest.approx(
            expected, abs=1e-6
        )


@pytest.mark.parametrize("size", [(11.0, 1.8), (20.0, 1.8), (100.0, 5.0), (40.0, 20.0)])
def test_bend_s_length_matches_gdsfactory(size):
    """Length of the 99-point Bezier polyline gf.components.bend_s draws."""
    dx, dy = size
    control_points = ((0, 0), (dx / 2, 0), (dx / 2, dy), (dx, dy))
    expected = _polyline_length(bezier_curve(np.linspace(0, 1, 99), control_points))
    assert float(_bend_s_length(dx, dy)) == pytest.approx(expected, abs=1e-9)
    component = gf.components.bend_s(
        size=size, cross_section="xs_sc340", allow_min_radius_violation=True
    )
    assert float(_bend_s_length(dx, dy)) == pytest.approx(
        component.info["length"], abs=1e-3
    )


def test_lengths_trace_under_jit():
    """Every argument of the length helpers can be a traced value."""
    euler = jax.jit(_euler_length)(30.0, 90.0, 0.5)
    sbend = jax.jit(_bend_s_length)(20.0, 1.8)
    assert float(euler) == pytest.approx(float(_euler_length(30.0, 90.0, 0.5)))
    assert float(sbend) == pytest.approx(float(_bend_s_length(20.0, 1.8)))
    grad = jax.grad(_euler_length)(30.0, 90.0, 0.5)
    assert np.isfinite(float(grad))


def test_optical_model_ports():
    """SAX in/out ports become one-based optical ports, in gdsfactory order."""
    mmi = _optical_model(sm.mmi1x2, 1, 2)
    ports = sax.get_ports(mmi(wl=jnp.array([1.55])))
    assert set(ports) == {"o1", "o2", "o3"}


def test_sdict_models_finds_annotated_public_callables():
    """Discovery keeps public ``-> SDict`` callables and partials of them."""

    def model(*, wl: float = 1.55) -> sax.SDict:
        return {}

    def other() -> int:
        return 0

    namespace = {
        "model": model,
        "alias": partial(model, wl=1.31),
        "other": other,
        "_private": model,
        "skipped": model,
    }
    assert set(_sdict_models(namespace, exclude={"skipped"})) == {"model", "alias"}
