"""Mach-Zehnder Interferometers (MZIs) are a type of interferometer used in optics.

They are used to measure the phase shift between two beams of light.

MZIs are used in a variety of applications, including optical communications, quantum computing, and sensing.
"""

import gdsfactory as gf
from gdsfactory.typings import ComponentSpec, CrossSectionSpec, Size

from cspdk.si220._schematic import FOUNDRY_SCHEMATICS as _FOUNDRY
from cspdk.si220._schematic import mzi_heater_loop_schematic, mzi_schematic
from cspdk.si220._utils import get_band_component as _get_band_component
from cspdk.si220.cells.heaters import heater
from cspdk.si220.cells.waveguides import bend_circular, straight


@gf.cell(tags=["mzis"], schematic_function=mzi_schematic)
def mzi(
    delta_length: float = 10,
    bend: ComponentSpec = "bend_euler",
    straight: ComponentSpec = "straight",
    splitter: ComponentSpec = "coupler",
    combiner: ComponentSpec | None = None,
    port_e1_splitter: str = "o3",
    port_e0_splitter: str = "o4",
    port_e1_combiner: str = "o3",
    port_e0_combiner: str = "o4",
    cross_section: CrossSectionSpec = "strip_cband",
) -> gf.Component:
    """Mzi.

    Args:
        delta_length: bottom arm vertical extra length.
        bend: 90 degrees bend library.
        straight: straight function.
        splitter: splitter function.
        combiner: combiner function.
        port_e1_splitter: east top splitter port.
        port_e0_splitter: east bot splitter port.
        port_e1_combiner: east top combiner port.
        port_e0_combiner: east bot combiner port.
        cross_section: for routing (sxtop/sxbot to combiner).
    """
    combiner = combiner or splitter
    _splitter = _get_band_component(splitter, cross_section)
    _combiner = _get_band_component(combiner, cross_section)
    if len(_splitter.ports) < 4:
        raise ValueError(
            f"Splitter {splitter} has {len(_splitter.ports)} ports, but needs at least 4 ports."
        )
    if len(_combiner.ports) < 4:
        raise ValueError(
            f"Combiner {combiner} has {len(_combiner.ports)} ports, but needs at least 4 ports."
        )

    return gf.c.mzi(
        delta_length=delta_length,
        bend=bend,
        straight=straight,
        splitter=_splitter,
        combiner=_combiner,
        port_e1_splitter=port_e1_splitter,
        port_e0_splitter=port_e0_splitter,
        port_e1_combiner=port_e1_combiner,
        port_e0_combiner=port_e0_combiner,
        cross_section=cross_section,
        length_y=2.0,
        length_x=0.1,
        straight_y=None,
        straight_x_top=None,
        straight_x_bot=None,
        with_splitter=True,
        port1="o1",
        port2="o2",
        nbends=2,
        cross_section_x_top=None,
        cross_section_x_bot=None,
        mirror_bot=False,
        add_optical_ports_arms=False,
        min_length=0.01,
        auto_rename_ports=True,
    )


@gf.cell(tags=["mzis"])
def mzi_lattice(
    delta_length: float = 10.0,
    bend: ComponentSpec = "bend_euler",
    straight: ComponentSpec = "straight",
    splitter: ComponentSpec = "mmi1x2",
    combiner: ComponentSpec = "mmi2x2",
    cross_section: CrossSectionSpec = "strip_cband",
) -> gf.Component:
    """Return an MZI lattice stage with one input and two outputs.

    Args:
        delta_length: Difference in length between the two MZI arms.
        bend: Bend component specification.
        straight: Straight component specification.
        splitter: One-by-two input splitter component specification.
        combiner: Two-by-two output combiner component specification.
        cross_section: Cross section used to route the arms.
    """
    _splitter = _get_band_component(splitter, cross_section)
    _combiner = _get_band_component(combiner, cross_section)
    return gf.c.mzi(
        delta_length=delta_length,
        length_y=1.0,
        length_x=0.1,
        straight_y=None,
        straight_x_top=None,
        straight_x_bot=None,
        with_splitter=True,
        port_e1_splitter="o2",
        port_e0_splitter="o3",
        port_e1_combiner="o3",
        port_e0_combiner="o4",
        nbends=2,
        cross_section=cross_section,
        cross_section_x_top=None,
        cross_section_x_bot=None,
        mirror_bot=False,
        add_optical_ports_arms=False,
        min_length=0.01,
        auto_rename_ports=True,
        bend=bend,
        straight=straight,
        splitter=_splitter,
        combiner=_combiner,
    )


def _route(
    c: gf.Component,
    port: gf.Port,
    steps: list[str | float],
    bend: gf.Component,
    cross_section: CrossSectionSpec,
) -> gf.Port:
    """Follow ``steps`` from ``port``: "L"/"R" turn with ``bend``, numbers are straights."""
    for step in steps:
        if step in ("L", "R"):
            ref = c << bend
            entry, exit_ = ("o1", "o2") if step == "L" else ("o2", "o1")
            ref.connect(entry, port)
            port = ref.ports[exit_]
        elif step:
            ref = c << straight(length=step, cross_section=cross_section)
            ref.connect("o1", port)
            port = ref.ports["o2"]
    return port


@gf.cell(tags=["mzis"], schematic_function=mzi_heater_loop_schematic)
def mzi_heater_loop(
    delta_length: float = 160.0,
    length_x: float = 616.3,
    length_y: float = 35.0,
    radius: float = 30.0,
    npoints: int | None = 187,
    splitter: ComponentSpec = "mmi1x2",
    combiner: ComponentSpec = "mmi2x2",
    heater_length: float = 200.0,
    heater_pad_size: Size = (200.0, 100.0),
    heater_pad_pitch: float = 250.0,
    cross_section: CrossSectionSpec = "strip_cband",
) -> gf.Component:
    """MZI whose arms loop around the outside, with a heater on the upper arm.

    The layout of the Cornerstone MZI: the 1x2 splitter faces west with its input
    (o1) on the east, the 2x2 combiner sits ``length_x`` east of the splitter
    outputs with its free ports (o2 lower, o3 upper) facing west, and each arm
    leaves the splitter, loops round with four circular bends and enters the
    combiner from the east. The lower arm is ``delta_length`` longer. A heater
    filament runs along the middle of the upper arm's horizontal straight, with
    its pads (e1, e2) to the north.

    Args:
        delta_length: lower arm length minus upper arm length (um).
        length_x: horizontal arm straight, splitter outputs to combiner ports (um).
        length_y: vertical straight on the splitter side of the upper arm (um).
        radius: bend radius (um).
        npoints: points per 90 degree bend. Defaults to the foundry MZI's.
        splitter: 1x2 splitter.
        combiner: 2x2 combiner.
        heater_length: heater filament length (um).
        heater_pad_size: heater pad width and height (um).
        heater_pad_pitch: distance between the heater pad centres (um).
        cross_section: waveguide cross-section.
    """
    c = gf.Component()
    spl = c << _get_band_component(splitter, cross_section)
    spl.drotate(180)
    upper, lower = spl.ports["o3"], spl.ports["o2"]
    comb = c << _get_band_component(combiner, cross_section)
    comb.dmovex(upper.x + length_x - comb.ports["o3"].x)
    bend = bend_circular(
        radius=radius,
        npoints=npoints,
        cross_section=cross_section,
        allow_min_radius_violation=True,
    )

    y_top = upper.y + 2 * radius + length_y
    y_bottom = lower.y - 2 * radius - length_y - delta_length / 2
    right_top = y_top - 2 * radius - comb.ports["o3"].y
    right_bottom = comb.ports["o4"].y - 2 * radius - y_bottom
    _route(
        c,
        upper,
        ["R", length_y, "R", length_x, "R", right_top, "R"],
        bend,
        cross_section,
    )
    _route(
        c,
        lower,
        ["L", length_y + delta_length / 2, "L", length_x, "L", right_bottom, "L"],
        bend,
        cross_section,
    )

    filament = c << heater(
        length=heater_length,
        pad_size=heater_pad_size,
        pad_pitch=heater_pad_pitch,
        pad_taper_length=0,
    )
    filament.dmove((upper.x + (length_x - heater_length) / 2, y_top))

    c.add_port("o1", port=spl.ports["o1"])
    c.add_port("o2", port=comb.ports["o1"])
    c.add_port("o3", port=comb.ports["o2"])
    c.add_port("e1", port=filament.ports["e1"])
    c.add_port("e2", port=filament.ports["e2"])
    c.info["delta_length"] = delta_length
    return c


################
# Foundry cells
################


@gf.cell(
    tags=["mzis", "foundry"], schematic_function=_FOUNDRY["SOI220_cband_TE_strip_MZI"]
)
def SOI220_cband_TE_strip_MZI() -> gf.Component:
    """Cornerstone C-band strip MZI with heater, without its grating couplers.

    The Run-50 template cell also holds three grating couplers and the routes to
    them; this is its interferometer: MMIs, arms and heater.
    """
    return mzi_heater_loop(cross_section="strip_cband")


@gf.cell(
    tags=["mzis", "foundry"], schematic_function=_FOUNDRY["SOI220_cband_TE_rib_MZI"]
)
def SOI220_cband_TE_rib_MZI() -> gf.Component:
    """Cornerstone C-band rib MZI with heater, without its grating couplers.

    The Run-50 template cell also holds three grating couplers and the routes to
    them; this is its interferometer: MMIs, arms and heater.
    """
    return mzi_heater_loop(length_x=649.5, cross_section="rib_cband")
