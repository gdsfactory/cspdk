"""Tests for the Ge-on-Si sample cells and their circuit simulation."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from mycspdk.samples.waveguide_loop import loop_transmission, waveguide_loop


def test_waveguide_loop_builds() -> None:
    """The loop has its two optical ports."""
    assert {p.name for p in waveguide_loop().ports} == {"o1", "o2"}


def test_loss_grows_linearly_with_length() -> None:
    """Doubling the arm length adds the same loss again: a clean cut-back."""
    loss = [
        -10 * float(jnp.log10(jnp.mean(loop_transmission(length=length)[1])))
        for length in (1000.0, 2000.0, 3000.0)
    ]
    assert 0 < loss[0] < loss[1] < loss[2]
    assert loss[2] - loss[1] == pytest.approx(loss[1] - loss[0], rel=1e-3)
