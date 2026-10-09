"""Functions for creating physical text geometries."""

from typing import Any

import gdsfactory as gf
from gdsfactory.typings import ComponentSpec, LayerSpec, LayerSpecs


@gf.cell(tags=["text"])
def text_rectangular(
    text: str = "abc",
    size: float = 3,
    justify: str = "left",
    layer: LayerSpec = "LBL",
) -> gf.Component:
    """Pixel based font, guaranteed to be manhattan, without acute angles.

    The default layer is the label layer (GDS 100), which Cornerstone merges into
    the dark-field silicon etch; its rules (250 nm width and gap) suit the font.
    Heater pads (GDS 41) need 10 um gaps, i.e. ``size`` >= 10 for text on PAD.

    Args:
        text: string.
        size: pixel size.
        justify: left, right or center.
        layer: for text.
    """
    return gf.c.text_rectangular(
        text=text, size=size, justify=justify, position=(0.0, 0.0), layer=layer
    )


@gf.cell(tags=["text"])
def text_rectangular_multi_layer(
    text: str = "abc",
    layers: LayerSpecs = ("WG", "SLAB"),
    text_factory: ComponentSpec = "text_rectangular",
    **kwargs: Any,
) -> gf.Component:
    """Returns rectangular text in different layers.

    The default draws the text as a rib (waveguide and slab layers), which passes
    the Cornerstone deck at the default 3 um pixel size.

    Args:
        text: string of text.
        layers: list of layers to replicate the text.
        text_factory: function to create the text Components.
        kwargs: keyword arguments for text_factory.
    """
    return gf.c.text_rectangular_multi_layer(
        text=text,
        layers=layers,
        text_factory=text_factory,
        **kwargs,
    )
