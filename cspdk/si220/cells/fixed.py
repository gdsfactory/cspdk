"""Fixed foundry cells imported from the Cornerstone MPW Run 50 template."""

import gdsfactory as gf

from cspdk.si220._schematic import (
    SOI220_cband_TE_strip_1x2_Ysplitter_schematic,
    SOI220_oband_TE_strip_1x2_Ysplitter_schematic,
)
from cspdk.si220.config import PATH

################
# Imported from the CORNERSTONE MPW Run 50 GDSII template
################
# Each cell is the foundry cell of the same name, extracted unchanged by
# scripts/extract_si220_run50.py. The port positions were measured on the template
# geometry; tests/test_si220_run50_fixed.py checks every optical port against the
# waveguide end it sits on.
#
# Waveguide ports are named o1, o2, ... clockwise from the west (the gdsfactory
# convention, as in mmi1x2), and grating-coupler fibre ports (``vertical_te``)
# follow them, as in the grating couplers (o1 waveguide, o2 fibre). A fibre port
# sits at the centre of the grating teeth and points away from the grating's
# waveguide. Electrical ports are named e1, e2, ... clockwise from the west and sit
# on the outer edge of each PAD (41/0) contact pad.

_FIBER_PORT_WIDTH = 10.0  # um, fibre mode-field diameter


def _import_run50(name: str) -> gf.Component:
    """Import the extracted Run 50 foundry cell ``name``."""
    return gf.import_gds(PATH.gds / f"{name}.gds")


def _add_optical_port(
    c: gf.Component,
    name: str,
    center: tuple[float, float],
    orientation: float,
    cross_section: str = "strip_cband",
) -> None:
    """Add a waveguide port with the width and layer of ``cross_section``."""
    c.add_port(
        name, center=center, orientation=orientation, cross_section=cross_section
    )


def _add_fiber_port(
    c: gf.Component, name: str, center: tuple[float, float], orientation: float
) -> None:
    """Add a grating-coupler fibre port at the centre of the grating teeth."""
    c.add_port(
        name,
        center=center,
        orientation=orientation,
        width=_FIBER_PORT_WIDTH,
        layer="GRA",
        port_type="vertical_te",
    )


def _add_pad_port(
    c: gf.Component,
    name: str,
    center: tuple[float, float],
    orientation: float,
    width: float,
) -> None:
    """Add an electrical port on the outer edge of a PAD contact pad."""
    c.add_port(
        name,
        center=center,
        orientation=orientation,
        width=width,
        layer="PAD",
        port_type="electrical",
    )


# Y-splitters


@gf.cell(
    tags=["fixed", "splitter", "1x2"],
    schematic_function=SOI220_cband_TE_strip_1x2_Ysplitter_schematic,
)
def SOI220_cband_TE_strip_1x2_Ysplitter() -> gf.Component:
    """C-band 1x2 Y-splitter for 450 nm strip waveguides (40 um x 20 um).

    o1 is the input at (0, 0); o2 and o3 are the outputs at (40, 10) and (40, -10).
    """
    c = _import_run50("SOI220_cband_TE_strip_1x2_Ysplitter")
    _add_optical_port(c, "o1", (0, 0), 180, "strip_cband")
    _add_optical_port(c, "o2", (40, 10), 0, "strip_cband")
    _add_optical_port(c, "o3", (40, -10), 0, "strip_cband")
    return c


@gf.cell(
    tags=["fixed", "splitter", "1x2"],
    schematic_function=SOI220_oband_TE_strip_1x2_Ysplitter_schematic,
)
def SOI220_oband_TE_strip_1x2_Ysplitter() -> gf.Component:
    """O-band 1x2 Y-splitter for 400 nm strip waveguides (40 um x 20 um).

    o1 is the input at (0, 0); o2 and o3 are the outputs at (40, 10) and (40, -10).
    """
    c = _import_run50("SOI220_oband_TE_strip_1x2_Ysplitter")
    _add_optical_port(c, "o1", (0, 0), 180, "strip_oband")
    _add_optical_port(c, "o2", (40, 10), 0, "strip_oband")
    _add_optical_port(c, "o3", (40, -10), 0, "strip_oband")
    return c


# Grating coupler


@gf.cell(tags=["fixed", "grating_couplers"])
def SOI220_cband_TE_strip_apodized_GC_focusing_v1p0() -> gf.Component:
    """C-band apodized focusing grating coupler for 450 nm strip waveguides.

    The grating lies west of the waveguide port o1 at (0, 0), which points east.
    The fibre port o2 is at the centre of the teeth, (-42.351, 0).
    """
    c = _import_run50("SOI220_cband_TE_strip_apodized_GC_focusing_v1p0")
    _add_optical_port(c, "o1", (0, 0), 0)
    _add_fiber_port(c, "o2", (-42.351, 0), 180)
    c.info["polarization"] = "te"
    c.info["wavelength"] = 1.55
    return c


# Graphene (single-layer graphene, SLG) devices
# A 450 nm strip waveguide runs along y through the device: o1 is its top end and
# o2 its bottom end. In the _with_GC cells it ends in a grating coupler at each end
# instead, so o1 and o2 are the fibre ports of the top and bottom gratings. The
# devices are probed on their Au/Ti pads (GRAPHENE_PAD1/2): the design rules keep
# PAD (41/0) 50 um away from the graphene exclusion zone, so there are no
# electrical ports.


@gf.cell(tags=["fixed", "graphene", "detector"])
def SOI220_cband_TE_strip_SLG_Bolometer() -> gf.Component:
    """Gated graphene bolometric photodetector (310 um x 320 um)."""
    c = _import_run50("SOI220_cband_TE_strip_SLG_Bolometer")
    _add_optical_port(c, "o1", (0, 150), 90)
    _add_optical_port(c, "o2", (0, -150), 270)
    return c


@gf.cell(tags=["fixed", "graphene", "detector"])
def SOI220_cband_TE_strip_SLG_Bolometer_with_GC() -> gf.Component:
    """Gated graphene bolometric photodetector between two grating couplers."""
    c = _import_run50("SOI220_cband_TE_strip_SLG_Bolometer_with_GC")
    _add_fiber_port(c, "o1", (0, 584.085), 90)
    _add_fiber_port(c, "o2", (0, -574.085), 270)
    return c


@gf.cell(tags=["fixed", "graphene", "detector"])
def SOI220_cband_TE_strip_SLG_Bolometer_metallic() -> gf.Component:
    """Graphene bolometric photodetector with a metallic gate (310 um x 310 um)."""
    c = _import_run50("SOI220_cband_TE_strip_SLG_Bolometer_metallic")
    _add_optical_port(c, "o1", (0, 150), 90)
    _add_optical_port(c, "o2", (0, -150), 270)
    return c


@gf.cell(tags=["fixed", "graphene", "detector"])
def SOI220_cband_TE_strip_SLG_Bolometer_metallic_with_GC() -> gf.Component:
    """Metallic-gate graphene bolometer between two grating couplers."""
    c = _import_run50("SOI220_cband_TE_strip_SLG_Bolometer_metallic_with_GC")
    _add_fiber_port(c, "o1", (0, 574.085), 90)
    _add_fiber_port(c, "o2", (0, -574.085), 270)
    return c


@gf.cell(tags=["fixed", "graphene", "detector"])
def SOI220_cband_TE_strip_SLG_Bolometer_no_gate() -> gf.Component:
    """Graphene bolometric photodetector without a gate (310 um x 320 um)."""
    c = _import_run50("SOI220_cband_TE_strip_SLG_Bolometer_no_gate")
    _add_optical_port(c, "o1", (0, 150), 90)
    _add_optical_port(c, "o2", (0, -150), 270)
    return c


@gf.cell(tags=["fixed", "graphene", "detector"])
def SOI220_cband_TE_strip_SLG_Bolometer_no_gate_with_GC() -> gf.Component:
    """Ungated graphene bolometer between two grating couplers."""
    c = _import_run50("SOI220_cband_TE_strip_SLG_Bolometer_no_gate_with_GC")
    _add_fiber_port(c, "o1", (0, 574.085), 90)
    _add_fiber_port(c, "o2", (0, -574.085), 270)
    return c


@gf.cell(tags=["fixed", "graphene", "modulator"])
def SOI220_cband_TE_strip_SLG_EAM() -> gf.Component:
    """Double-layer graphene electro-absorption modulator (210 um x 210 um)."""
    c = _import_run50("SOI220_cband_TE_strip_SLG_EAM")
    _add_optical_port(c, "o1", (0, 100), 90)
    _add_optical_port(c, "o2", (0, -100), 270)
    return c


@gf.cell(tags=["fixed", "graphene", "modulator"])
def SOI220_cband_TE_strip_SLG_EAM_with_GC() -> gf.Component:
    """Graphene electro-absorption modulator between two grating couplers."""
    c = _import_run50("SOI220_cband_TE_strip_SLG_EAM_with_GC")
    _add_fiber_port(c, "o1", (0, 524.085), 90)
    _add_fiber_port(c, "o2", (0, -524.085), 270)
    return c


@gf.cell(tags=["fixed", "graphene", "detector"])
def SOI220_cband_TE_strip_SLG_PTE() -> gf.Component:
    """Graphene photo-thermoelectric detector (210 um x 230 um)."""
    c = _import_run50("SOI220_cband_TE_strip_SLG_PTE")
    _add_optical_port(c, "o1", (0, 110), 90)
    _add_optical_port(c, "o2", (0, -110), 270)
    return c


@gf.cell(tags=["fixed", "graphene", "detector"])
def SOI220_cband_TE_strip_SLG_PTE_with_GC() -> gf.Component:
    """Graphene photo-thermoelectric detector between two grating couplers."""
    c = _import_run50("SOI220_cband_TE_strip_SLG_PTE_with_GC")
    _add_fiber_port(c, "o1", (0, 534.085), 90)
    _add_fiber_port(c, "o2", (0, -534.085), 270)
    return c


@gf.cell(tags=["fixed", "graphene", "detector"])
def SOI220_cband_TE_strip_SLG_PTE_metallic() -> gf.Component:
    """Metallic-gate graphene photo-thermoelectric detector (210 um x 230 um)."""
    c = _import_run50("SOI220_cband_TE_strip_SLG_PTE_metallic")
    _add_optical_port(c, "o1", (0, 110), 90)
    _add_optical_port(c, "o2", (0, -110), 270)
    return c


@gf.cell(tags=["fixed", "graphene", "detector"])
def SOI220_cband_TE_strip_SLG_PTE_metallic_with_GC() -> gf.Component:
    """Metallic-gate graphene photo-thermoelectric detector between two gratings."""
    c = _import_run50("SOI220_cband_TE_strip_SLG_PTE_metallic_with_GC")
    _add_fiber_port(c, "o1", (0, 534.085), 90)
    _add_fiber_port(c, "o2", (0, -534.085), 270)
    return c


# Flip-chip photodiode (OSI Optoelectronics FCI-InGaAs-300B1)
# The 300 um active area is centred on (0, 0). e1 is on the top edge of the cathode
# pad and e2 on the bottom edge of the anode pad.


def _add_photodiode_pads(c: gf.Component) -> None:
    """Add the cathode (e1) and anode (e2) pad ports of the photodiode."""
    _add_pad_port(c, "e1", (0, 325), 90, 202)
    _add_pad_port(c, "e2", (0, -315), 270, 300)


@gf.cell(tags=["fixed", "photodiode", "detector"])
def SOI220_PD_FCI300B1_Plain() -> gf.Component:
    """Flip-chip InGaAs photodiode site with its contact pads and no waveguide."""
    c = _import_run50("SOI220_PD_FCI300B1_Plain")
    _add_photodiode_pads(c)
    return c


@gf.cell(tags=["fixed", "photodiode", "detector"])
def SOI220_PD_FCI300B1_Integrated() -> gf.Component:
    """Flip-chip InGaAs photodiode fed by a grating coupler under its active area.

    Light enters through the strip waveguide port o1 at the east edge, (537, 0).
    """
    c = _import_run50("SOI220_PD_FCI300B1_Integrated")
    _add_optical_port(c, "o1", (537, 0), 0)
    _add_photodiode_pads(c)
    return c


# Integrated laser (Shijia Photons DC1550HP01)
# Each cell is a complete Full (11.47 mm x 4.9 mm) or Half (5.5 mm x 4.9 mm) user
# cell with the laser site on its west side. The laser edge-couples into a taper
# whose 450 nm end, o1, is the laser output at the edge of the laser exclusion
# zone. In _or_GC a focusing grating coupler (fibre port o2) on the laser site
# feeds o1 instead of a laser. In _with_GC a 2x1 combiner joins the laser and a
# focusing grating coupler (fibre port o2) north of it into o1. The laser contacts
# are probed directly, so there are no electrical ports.


@gf.cell(tags=["fixed", "laser"])
def SOI220_Full_LD_DC1550HP01_Integrated() -> gf.Component:
    """Full user cell with an integrated 1550 nm laser feeding o1."""
    c = _import_run50("SOI220_Full_LD_DC1550HP01_Integrated")
    _add_optical_port(c, "o1", (-3456.39, 0), 0)
    return c


@gf.cell(tags=["fixed", "laser"])
def SOI220_Full_LD_DC1550HP01_Integrated_or_GC() -> gf.Component:
    """Full user cell with a focusing grating coupler on the laser site feeding o1."""
    c = _import_run50("SOI220_Full_LD_DC1550HP01_Integrated_or_GC")
    _add_optical_port(c, "o1", (-3456.39, 0), 0)
    _add_fiber_port(c, "o2", (-3977.351, 0), 180)
    return c


@gf.cell(tags=["fixed", "laser"])
def SOI220_Full_LD_DC1550HP01_Integrated_with_GC() -> gf.Component:
    """Full user cell with an integrated laser and a grating coupler feeding o1."""
    c = _import_run50("SOI220_Full_LD_DC1550HP01_Integrated_with_GC")
    _add_optical_port(c, "o1", (-3456.39, 1.57), 0)
    _add_fiber_port(c, "o2", (-3630.541, 800), 180)
    return c


@gf.cell(tags=["fixed", "laser"])
def SOI220_Half_LD_DC1550HP01_Integrated() -> gf.Component:
    """Half user cell with an integrated 1550 nm laser feeding o1."""
    c = _import_run50("SOI220_Half_LD_DC1550HP01_Integrated")
    _add_optical_port(c, "o1", (-471.39, 0), 0)
    return c


@gf.cell(tags=["fixed", "laser"])
def SOI220_Half_LD_DC1550HP01_Integrated_or_GC() -> gf.Component:
    """Half user cell with a focusing grating coupler on the laser site feeding o1."""
    c = _import_run50("SOI220_Half_LD_DC1550HP01_Integrated_or_GC")
    _add_optical_port(c, "o1", (-471.39, 0), 0)
    _add_fiber_port(c, "o2", (-992.351, 0), 180)
    return c


@gf.cell(tags=["fixed", "laser"])
def SOI220_Half_LD_DC1550HP01_Integrated_with_GC() -> gf.Component:
    """Half user cell with an integrated laser and a grating coupler feeding o1."""
    c = _import_run50("SOI220_Half_LD_DC1550HP01_Integrated_with_GC")
    _add_optical_port(c, "o1", (-471.39, 1.57), 0)
    _add_fiber_port(c, "o2", (-645.541, 800), 180)
    return c
