"""Tests for active circuit sample components."""

from __future__ import annotations

from mycspdk.samples.heater_iv_curve import plot_heater_iv
from mycspdk.samples.heater_resistance_vs_length import (
    plot_resistance_vs_length,
    show_heater_cells,
)
from mycspdk.samples.thermal_mzi_sweep import (
    plot_expected_transfer_function,
    thermal_mzi,
)


def test_thermal_mzi_cell() -> None:
    """Thermal MZI cell builds without error."""
    c = thermal_mzi()
    assert c.ports


def test_thermal_mzi_transfer_function() -> None:
    """Transfer function plot generates without error."""
    fig = plot_expected_transfer_function(n_points=100)
    assert fig is not None
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_resistance_vs_length_plot() -> None:
    """Resistance vs length plot generates without error."""
    fig = plot_resistance_vs_length(lengths=(100.0, 320.0))
    assert fig is not None
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_heater_cells_different_lengths() -> None:
    """Heater cells at different lengths build without error."""
    c = show_heater_cells(lengths=(100.0, 320.0))
    assert c is not None


def test_heater_iv_plot() -> None:
    """IV curve plot generates without error."""
    fig = plot_heater_iv(n_points=100)
    assert fig is not None
    import matplotlib.pyplot as plt

    plt.close(fig)
