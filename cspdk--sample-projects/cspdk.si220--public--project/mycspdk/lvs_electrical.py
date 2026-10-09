"""LVS demo."""

import gdsfactory as gf
from cspdk.si220 import cells


def _add_wired_pads(c: gf.Component, pad, cross_section) -> dict[str, gf.Instance]:
    """Place four labelled pads and wire each left pad to the right pad facing it."""
    layer = gf.get_layer(gf.get_cross_section(cross_section).layer)
    pad = gf.get_component(pad)

    tl = c << pad
    bl = c << pad
    tr = c << pad
    br = c << pad
    tl.move((0, 300))
    br.move((500, 0))
    tr.move((500, 500))

    pads = {"tl": tl, "tr": tr, "br": br, "bl": bl}
    for name, ref in pads.items():
        ref.name = name
        c.add_label(name, position=ref.dcenter, layer=layer)

    ports1 = [bl.ports["e3"], tl.ports["e3"]]
    ports2 = [br.ports["e1"], tr.ports["e1"]]
    gf.routing.route_bundle_electrical(c, ports1, ports2, cross_section=cross_section)
    return pads


@gf.cell
def pads_correct(pad=cells.pad, cross_section="metal_routing") -> gf.Component:
    """Returns 4 pads, wired pairwise left to right with metal wires."""
    c = gf.Component()
    _add_wired_pads(c, pad, cross_section)
    return c


@gf.cell
def pads_shorted(pad=cells.pad, cross_section="metal_routing") -> gf.Component:
    """Same as pads_correct, plus a wire shorting the two left pads together."""
    c = gf.Component()
    pads = _add_wired_pads(c, pad, cross_section)
    gf.routing.route_bundle_electrical(
        c,
        [pads["bl"].ports["e2"]],
        [pads["tl"].ports["e4"]],
        cross_section=cross_section,
    )
    return c
