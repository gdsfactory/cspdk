"""Mach-Zehnder Interferometers (MZIs) are a type of interferometer used in optics.

They are used to measure the phase shift between two beams of light.

MZIs are used in a variety of applications, including optical communications, quantum computing, and sensing.
"""

import gdsfactory as gf
from gdsfactory.typings import ComponentSpec, CrossSectionSpec

from cspdk.si220._schematic import mzi_schematic


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
    _splitter = gf.get_component(splitter)
    _combiner = gf.get_component(combiner)
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
        splitter=splitter,
        combiner=combiner,
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
        splitter=splitter,
        combiner=combiner,
    )
