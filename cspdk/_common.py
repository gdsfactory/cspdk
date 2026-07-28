"""Common utilities shared across cspdk PDK variants."""

from __future__ import annotations

from functools import partial

from gdsfactory.add_pins import add_electric_pins

# Logical-only electrical pin helper.
# pin_layer_map with (0,0)->None suppresses geometry while registering
# logical pins via create_pin(). Future TODO: map actual metal layers.
_add_pins = partial(add_electric_pins, pin_layer_map={(0, 0): None})
