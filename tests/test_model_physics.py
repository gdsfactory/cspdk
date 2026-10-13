"""Physical sanity of every SAX model of every flavour, swept across its band.

Each model is evaluated at its default parameters over +-3 % of its centre
wavelength (``wl``'s default). si220 models choose their band from the
cross-section, so each is also swept with the O-band version of its default
cross-section at 1.31 um. The S-matrix must be:

* finite: no NaN or Inf;
* passive: its largest singular value is at most 1, so no combination of inputs,
  coherent or not, comes out with more power than went in;
* reciprocal: ``S[i, j] == S[j, i]``, with both directions present.

Interpolated measured tables and fitted data can break these at the band edges,
which the model contract test (evaluation, jit, circulax) would not notice.
"""

from __future__ import annotations

import inspect

import jax.numpy as jnp
import numpy as np
import pytest
from test_model_contract import _flavour_models, _wl0

TOLERANCE = 1e-6
# Test ids ("<flavour>-<model>") of models that are non-reciprocal by design.
NON_RECIPROCAL: set[str] = set()
SI220_OBAND_WL = 1.31


def _cases():
    for flavour, name, model in _flavour_models():
        yield pytest.param(model, {}, _wl0(model), id=f"{flavour}-{name}")
        xs = inspect.signature(model).parameters.get("cross_section")
        if flavour == "si220" and xs and str(xs.default).endswith("_cband"):
            oband = {"cross_section": str(xs.default).replace("_cband", "_oband")}
            yield pytest.param(
                model, oband, SI220_OBAND_WL, id=f"{flavour}-{name}-oband"
            )


CASES = list(_cases())


def _s_matrix(model, kwargs, wl0) -> tuple[list[str], np.ndarray]:
    """Ports and the S-matrix, shaped (wavelength, output, input)."""
    wl = jnp.linspace(0.97 * wl0, 1.03 * wl0, 41)
    s = model(wl=wl, **kwargs)
    ports = sorted({port for pair in s for port in pair})
    matrix = np.zeros((wl.size, len(ports), len(ports)), dtype=complex)
    for (i, j), value in s.items():
        matrix[:, ports.index(i), ports.index(j)] = np.broadcast_to(value, wl.shape)
    return ports, matrix


@pytest.mark.parametrize(("model", "kwargs", "wl0"), CASES)
def test_model_is_finite(model, kwargs, wl0) -> None:
    """No S-parameter is NaN or Inf anywhere in the band."""
    _, s = _s_matrix(model, kwargs, wl0)
    assert np.all(np.isfinite(s))


@pytest.mark.parametrize(("model", "kwargs", "wl0"), CASES)
def test_model_is_passive(model, kwargs, wl0) -> None:
    """No combination of inputs comes out with more power than went in."""
    _, s = _s_matrix(model, kwargs, wl0)
    gain = np.linalg.svd(s, compute_uv=False).max(axis=-1) ** 2
    assert np.max(gain) <= 1 + TOLERANCE


@pytest.mark.parametrize(("model", "kwargs", "wl0"), CASES)
def test_model_is_reciprocal(model, kwargs, wl0, request) -> None:
    """S[i, j] equals S[j, i], and every entry has its reverse direction."""
    if request.node.callspec.id in NON_RECIPROCAL:
        pytest.skip("non-reciprocal by design")
    s = model(wl=jnp.linspace(0.97 * wl0, 1.03 * wl0, 41), **kwargs)
    for (i, j), value in s.items():
        assert (j, i) in s, (i, j)
        np.testing.assert_allclose(value, s[(j, i)], atol=TOLERANCE, err_msg=f"{i},{j}")
