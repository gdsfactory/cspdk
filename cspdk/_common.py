"""Common utilities shared across cspdk PDK variants."""

from __future__ import annotations

from gdsfactory.add_pins import add_electric_pins

# All active Cornerstone variants share these electrical drawing layers.
_ELECTRICAL_DRAWING_LAYERS = (
    (39, 0),  # HEATER — TiN heater layer
    (41, 0),  # PAD    — bond-pad metal
)


def _add_pins(component) -> None:
    """Register logical electrical pins; geometric pin drawing disabled pending reference GDS update."""
    add_electric_pins(
        component,
        pin_layer_map={
            component.kcl.layer(*s): None for s in _ELECTRICAL_DRAWING_LAYERS
        },
    )
