"""Physical sanity of every SAX model of every flavour, swept across its band.

Each model is evaluated at its default parameters over +-3 % of its centre
wavelength (``wl``'s default). The S-matrix must be:

* finite: no NaN or Inf;
* passive: for each input port, the total output power is at most 1;
* reciprocal: ``S[i, j] == S[j, i]``, with both directions present.

Interpolated measured tables and fitted data can break these at the band edges,
which the model contract test (evaluation, jit, circulax) would not notice.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from test_model_contract import _sax_models, _wl0

TOLERANCE = 1e-6
# Models that are non-reciprocal by design would be listed here, by function name.
NON_RECIPROCAL: set[str] = set()


def _s_matrix(model) -> tuple[list[str], dict[tuple[str, str], np.ndarray]]:
    wl0 = _wl0(model)
    wl = jnp.linspace(0.97 * wl0, 1.03 * wl0, 41)
    s = model(wl=wl)
    ports = sorted({port for pair in s for port in pair})
    return ports, {k: np.broadcast_to(np.asarray(v), wl.shape) for k, v in s.items()}


@pytest.mark.parametrize("model", _sax_models())
def test_model_is_finite(model) -> None:
    """No S-parameter is NaN or Inf anywhere in the band."""
    _, s = _s_matrix(model)
    for pair, value in s.items():
        assert np.all(np.isfinite(value)), pair


@pytest.mark.parametrize("model", _sax_models())
def test_model_is_passive(model) -> None:
    """No input port sends out more power than it receives."""
    ports, s = _s_matrix(model)
    for j in ports:
        power = sum(np.abs(s.get((i, j), 0.0)) ** 2 for i in ports)
        assert np.max(power) <= 1 + TOLERANCE, j


@pytest.mark.parametrize("model", _sax_models())
def test_model_is_reciprocal(model) -> None:
    """S[i, j] equals S[j, i], and every entry has its reverse direction."""
    if getattr(model, "__name__", None) in NON_RECIPROCAL:
        pytest.skip("non-reciprocal by design")
    _, s = _s_matrix(model)
    for (i, j), value in s.items():
        assert (j, i) in s, (i, j)
        np.testing.assert_allclose(value, s[(j, i)], atol=TOLERANCE, err_msg=f"{i},{j}")
