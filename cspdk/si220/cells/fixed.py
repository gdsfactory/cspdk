"""Fixed foundry cells imported from the Cornerstone MPW Run 50 template."""

import gdsfactory as gf
from gdsfactory.typings import CrossSectionSpec

from cspdk.si220._schematic import ysplitter_schematic
from cspdk.si220.config import PATH
from cspdk.si220.tech import get_band, is_rib

################
# Imported from the CORNERSTONE MPW Run 50 GDSII template
################
# Each cell is a foundry cell, named in its docstring, extracted unchanged by
# scripts/extract_si220_run50.py into the GDS file of the foundry cell's name. The
# port positions were measured on the template geometry;
# tests/test_si220_run50_fixed.py checks every optical port against the waveguide
# end it sits on.
#
# Waveguide ports are named o1, o2, ... clockwise from the west (the gdsfactory
# convention, as in mmi1x2), and grating-coupler fibre ports (``vertical_te``)
# follow them, as in the grating couplers (o1 waveguide, o2 fibre). A fibre port
# sits at the centre of the grating teeth and points away from the grating's
# waveguide. Electrical ports are named e1, e2, ... clockwise from the west and sit
# on the outer edge of each contact pad, on the pad's own layer: PAD (41/0), or
# GRAPHENE_PAD1/2 (165/2, 166/2) for the graphene devices.

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
    layer: str = "PAD",
) -> None:
    """Add an electrical port on the outer edge of a contact pad on ``layer``."""
    c.add_port(
        name,
        center=center,
        orientation=orientation,
        width=width,
        layer=layer,
        port_type="electrical",
    )


# Y-splitter


@gf.cell(tags=["fixed", "splitter", "1x2"], schematic_function=ysplitter_schematic)
def ysplitter(cross_section: CrossSectionSpec = "strip_cband") -> gf.Component:
    """1x2 Y-splitter (40 um x 20 um) for the cross-section's band.

    The Cornerstone Run 50 cell ``SOI220_cband_TE_strip_1x2_Ysplitter`` (450 nm
    strip) or ``SOI220_oband_TE_strip_1x2_Ysplitter`` (400 nm strip). o1 is the
    input at (0, 0); o2 and o3 are the outputs at (40, 10) and (40, -10).

    Args:
        cross_section: strip C- or O-band cross-section. The foundry has no rib
            Y-splitter.
    """
    if is_rib(cross_section):
        raise ValueError(
            "ysplitter is a strip cell (the foundry has no rib Y-splitter); "
            "use a strip cross-section such as 'strip_cband' or 'strip_oband'"
        )
    band = get_band(cross_section)
    strip = f"strip_{band}"
    c = _import_run50(f"SOI220_{band}_TE_strip_1x2_Ysplitter")
    _add_optical_port(c, "o1", (0, 0), 180, strip)
    _add_optical_port(c, "o2", (40, 10), 0, strip)
    _add_optical_port(c, "o3", (40, -10), 0, strip)
    return c


# Grating coupler


@gf.cell(tags=["fixed", "grating_couplers"])
def grating_coupler_focusing_apodized() -> gf.Component:
    """C-band apodized focusing grating coupler for 450 nm strip waveguides.

    Cornerstone Run 50 cell ``SOI220_cband_TE_strip_apodized_GC_focusing_v1p0``.
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
# o2 its bottom end. In the _with_gc cells it ends in a grating coupler at each end
# instead, so o1 and o2 are the fibre ports of the top and bottom gratings; the
# device and its pads are at the same place as in the bare cell.
#
# The devices are probed on their Au/Ti pads, GRAPHENE_PAD1 (165/2) and
# GRAPHENE_PAD2 (166/2), labelled S (signal), G (ground) and Gate in the
# standard-components datasheet (pp. 30-35). The design rules keep PAD (41/0)
# 50 um away from the graphene exclusion zone (169/1), so the pad ports are on the
# pad's own layer and nothing is drawn on PAD. Every pad lies beside the
# waveguide (x = 0), so each port is on the pad edge facing away from it: west of
# the waveguide the ports point west, east of it they point east. The pad
# positions were measured on the template (tests/test_si220_run50_fixed.py).


def _add_bolometer_pads(c: gf.Component, gate: bool = True) -> None:
    """Add the S (e1), Gate (e2) and G (e3, or e2 without a gate) pad ports.

    The S and G pads (GRAPHENE_PAD1, 60 um x 90 um) contact the graphene channel
    either side of the waveguide; the Gate pad (GRAPHENE_PAD2) is above G.
    """
    _add_pad_port(c, "e1", (-65, 0), 180, 90, "GRAPHENE_PAD1")
    if gate:
        _add_pad_port(c, "e2", (65, 100), 0, 90, "GRAPHENE_PAD2")
    _add_pad_port(c, "e3" if gate else "e2", (65, 0), 0, 90, "GRAPHENE_PAD1")


def _add_eam_pads(c: gf.Component) -> None:
    """Add the S (e1, 1st graphene) and G (e2, 2nd graphene) pad ports.

    The two 60 um x 90 um pads contact the two graphene layers of the capacitor.
    """
    _add_pad_port(c, "e1", (-65, 0), 180, 90, "GRAPHENE_PAD1")
    _add_pad_port(c, "e2", (65, 0), 0, 90, "GRAPHENE_PAD2")


def _add_pte_pads(c: gf.Component) -> None:
    """Add the S (e1), Gate1 (e2), Gate2 (e3) and G (e4) pad ports.

    The 60 um x 60 um S and G pads (GRAPHENE_PAD1) contact the graphene channel;
    the split-gate pads Gate1 and Gate2 (GRAPHENE_PAD2) are above them.
    """
    _add_pad_port(c, "e1", (-65, 0), 180, 60, "GRAPHENE_PAD1")
    _add_pad_port(c, "e2", (-65, 70), 180, 60, "GRAPHENE_PAD2")
    _add_pad_port(c, "e3", (65, 70), 0, 60, "GRAPHENE_PAD2")
    _add_pad_port(c, "e4", (65, 0), 0, 60, "GRAPHENE_PAD1")


@gf.cell(tags=["fixed", "graphene", "detector"])
def graphene_bolometer() -> gf.Component:
    """Gated graphene bolometric photodetector (310 um x 320 um).

    Cornerstone Run 50 cell ``SOI220_cband_TE_strip_SLG_Bolometer``.
    """
    c = _import_run50("SOI220_cband_TE_strip_SLG_Bolometer")
    _add_optical_port(c, "o1", (0, 150), 90)
    _add_optical_port(c, "o2", (0, -150), 270)
    _add_bolometer_pads(c)
    return c


@gf.cell(tags=["fixed", "graphene", "detector"])
def graphene_bolometer_with_gc() -> gf.Component:
    """Gated graphene bolometric photodetector between two grating couplers.

    Cornerstone Run 50 cell ``SOI220_cband_TE_strip_SLG_Bolometer_with_GC``.
    """
    c = _import_run50("SOI220_cband_TE_strip_SLG_Bolometer_with_GC")
    _add_fiber_port(c, "o1", (0, 584.085), 90)
    _add_fiber_port(c, "o2", (0, -574.085), 270)
    _add_bolometer_pads(c)
    return c


@gf.cell(tags=["fixed", "graphene", "detector"])
def graphene_bolometer_metallic() -> gf.Component:
    """Graphene bolometric photodetector with a metallic gate (310 um x 310 um).

    Cornerstone Run 50 cell ``SOI220_cband_TE_strip_SLG_Bolometer_metallic``.
    """
    c = _import_run50("SOI220_cband_TE_strip_SLG_Bolometer_metallic")
    _add_optical_port(c, "o1", (0, 150), 90)
    _add_optical_port(c, "o2", (0, -150), 270)
    _add_bolometer_pads(c)
    return c


@gf.cell(tags=["fixed", "graphene", "detector"])
def graphene_bolometer_metallic_with_gc() -> gf.Component:
    """Metallic-gate graphene bolometer between two grating couplers.

    Cornerstone Run 50 cell ``SOI220_cband_TE_strip_SLG_Bolometer_metallic_with_GC``.
    """
    c = _import_run50("SOI220_cband_TE_strip_SLG_Bolometer_metallic_with_GC")
    _add_fiber_port(c, "o1", (0, 574.085), 90)
    _add_fiber_port(c, "o2", (0, -574.085), 270)
    _add_bolometer_pads(c)
    return c


@gf.cell(tags=["fixed", "graphene", "detector"])
def graphene_bolometer_no_gate() -> gf.Component:
    """Graphene bolometric photodetector without a gate (310 um x 320 um).

    Cornerstone Run 50 cell ``SOI220_cband_TE_strip_SLG_Bolometer_no_gate``.
    """
    c = _import_run50("SOI220_cband_TE_strip_SLG_Bolometer_no_gate")
    _add_optical_port(c, "o1", (0, 150), 90)
    _add_optical_port(c, "o2", (0, -150), 270)
    _add_bolometer_pads(c, gate=False)
    return c


@gf.cell(tags=["fixed", "graphene", "detector"])
def graphene_bolometer_no_gate_with_gc() -> gf.Component:
    """Ungated graphene bolometer between two grating couplers.

    Cornerstone Run 50 cell ``SOI220_cband_TE_strip_SLG_Bolometer_no_gate_with_GC``.
    """
    c = _import_run50("SOI220_cband_TE_strip_SLG_Bolometer_no_gate_with_GC")
    _add_fiber_port(c, "o1", (0, 574.085), 90)
    _add_fiber_port(c, "o2", (0, -574.085), 270)
    _add_bolometer_pads(c, gate=False)
    return c


@gf.cell(tags=["fixed", "graphene", "modulator"])
def graphene_eam() -> gf.Component:
    """Double-layer graphene electro-absorption modulator (210 um x 210 um).

    Cornerstone Run 50 cell ``SOI220_cband_TE_strip_SLG_EAM``.
    """
    c = _import_run50("SOI220_cband_TE_strip_SLG_EAM")
    _add_optical_port(c, "o1", (0, 100), 90)
    _add_optical_port(c, "o2", (0, -100), 270)
    _add_eam_pads(c)
    return c


@gf.cell(tags=["fixed", "graphene", "modulator"])
def graphene_eam_with_gc() -> gf.Component:
    """Graphene electro-absorption modulator between two grating couplers.

    Cornerstone Run 50 cell ``SOI220_cband_TE_strip_SLG_EAM_with_GC``.
    """
    c = _import_run50("SOI220_cband_TE_strip_SLG_EAM_with_GC")
    _add_fiber_port(c, "o1", (0, 524.085), 90)
    _add_fiber_port(c, "o2", (0, -524.085), 270)
    _add_eam_pads(c)
    return c


@gf.cell(tags=["fixed", "graphene", "detector"])
def graphene_pte() -> gf.Component:
    """Graphene photo-thermoelectric detector (210 um x 230 um).

    Cornerstone Run 50 cell ``SOI220_cband_TE_strip_SLG_PTE``.
    """
    c = _import_run50("SOI220_cband_TE_strip_SLG_PTE")
    _add_optical_port(c, "o1", (0, 110), 90)
    _add_optical_port(c, "o2", (0, -110), 270)
    _add_pte_pads(c)
    return c


@gf.cell(tags=["fixed", "graphene", "detector"])
def graphene_pte_with_gc() -> gf.Component:
    """Graphene photo-thermoelectric detector between two grating couplers.

    Cornerstone Run 50 cell ``SOI220_cband_TE_strip_SLG_PTE_with_GC``.
    """
    c = _import_run50("SOI220_cband_TE_strip_SLG_PTE_with_GC")
    _add_fiber_port(c, "o1", (0, 534.085), 90)
    _add_fiber_port(c, "o2", (0, -534.085), 270)
    _add_pte_pads(c)
    return c


@gf.cell(tags=["fixed", "graphene", "detector"])
def graphene_pte_metallic() -> gf.Component:
    """Metallic-gate graphene photo-thermoelectric detector (210 um x 230 um).

    Cornerstone Run 50 cell ``SOI220_cband_TE_strip_SLG_PTE_metallic``.
    """
    c = _import_run50("SOI220_cband_TE_strip_SLG_PTE_metallic")
    _add_optical_port(c, "o1", (0, 110), 90)
    _add_optical_port(c, "o2", (0, -110), 270)
    _add_pte_pads(c)
    return c


@gf.cell(tags=["fixed", "graphene", "detector"])
def graphene_pte_metallic_with_gc() -> gf.Component:
    """Metallic-gate graphene photo-thermoelectric detector between two gratings.

    Cornerstone Run 50 cell ``SOI220_cband_TE_strip_SLG_PTE_metallic_with_GC``.
    """
    c = _import_run50("SOI220_cband_TE_strip_SLG_PTE_metallic_with_GC")
    _add_fiber_port(c, "o1", (0, 534.085), 90)
    _add_fiber_port(c, "o2", (0, -534.085), 270)
    _add_pte_pads(c)
    return c


# Flip-chip photodiode (OSI Optoelectronics FCI-InGaAs-300B1)
# The 300 um active area is centred on (0, 0). e1 is on the top edge of the cathode
# pad and e2 on the bottom edge of the anode pad.


def _add_photodiode_pads(c: gf.Component) -> None:
    """Add the cathode (e1) and anode (e2) pad ports of the photodiode."""
    _add_pad_port(c, "e1", (0, 325), 90, 202)
    _add_pad_port(c, "e2", (0, -315), 270, 300)


@gf.cell(tags=["fixed", "photodiode", "detector"])
def photodiode() -> gf.Component:
    """Flip-chip InGaAs photodiode site with its contact pads and no waveguide.

    Cornerstone Run 50 cell ``SOI220_PD_FCI300B1_Plain``.
    """
    c = _import_run50("SOI220_PD_FCI300B1_Plain")
    _add_photodiode_pads(c)
    return c


@gf.cell(tags=["fixed", "photodiode", "detector"])
def photodiode_integrated() -> gf.Component:
    """Flip-chip InGaAs photodiode fed by a grating coupler under its active area.

    Cornerstone Run 50 cell ``SOI220_PD_FCI300B1_Integrated``.
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
# zone. In _or_gc a focusing grating coupler (fibre port o2) on the laser site
# feeds o1 instead of a laser. In _with_gc a 2x1 combiner joins the laser and a
# focusing grating coupler (fibre port o2) north of it into o1. The laser contacts
# are probed directly, so there are no electrical ports.


@gf.cell(tags=["fixed", "laser"])
def laser_full() -> gf.Component:
    """Full user cell with an integrated 1550 nm laser feeding o1.

    Cornerstone Run 50 cell ``SOI220_Full_LD_DC1550HP01_Integrated``.
    """
    c = _import_run50("SOI220_Full_LD_DC1550HP01_Integrated")
    _add_optical_port(c, "o1", (-3456.39, 0), 0)
    return c


@gf.cell(tags=["fixed", "laser"])
def laser_full_or_gc() -> gf.Component:
    """Full user cell with a focusing grating coupler on the laser site feeding o1.

    Cornerstone Run 50 cell ``SOI220_Full_LD_DC1550HP01_Integrated_or_GC``.
    """
    c = _import_run50("SOI220_Full_LD_DC1550HP01_Integrated_or_GC")
    _add_optical_port(c, "o1", (-3456.39, 0), 0)
    _add_fiber_port(c, "o2", (-3977.351, 0), 180)
    return c


@gf.cell(tags=["fixed", "laser"])
def laser_full_with_gc() -> gf.Component:
    """Full user cell with an integrated laser and a grating coupler feeding o1.

    Cornerstone Run 50 cell ``SOI220_Full_LD_DC1550HP01_Integrated_with_GC``.
    """
    c = _import_run50("SOI220_Full_LD_DC1550HP01_Integrated_with_GC")
    _add_optical_port(c, "o1", (-3456.39, 1.57), 0)
    _add_fiber_port(c, "o2", (-3630.541, 800), 180)
    return c


@gf.cell(tags=["fixed", "laser"])
def laser_half() -> gf.Component:
    """Half user cell with an integrated 1550 nm laser feeding o1.

    Cornerstone Run 50 cell ``SOI220_Half_LD_DC1550HP01_Integrated``.
    """
    c = _import_run50("SOI220_Half_LD_DC1550HP01_Integrated")
    _add_optical_port(c, "o1", (-471.39, 0), 0)
    return c


@gf.cell(tags=["fixed", "laser"])
def laser_half_or_gc() -> gf.Component:
    """Half user cell with a focusing grating coupler on the laser site feeding o1.

    Cornerstone Run 50 cell ``SOI220_Half_LD_DC1550HP01_Integrated_or_GC``.
    """
    c = _import_run50("SOI220_Half_LD_DC1550HP01_Integrated_or_GC")
    _add_optical_port(c, "o1", (-471.39, 0), 0)
    _add_fiber_port(c, "o2", (-992.351, 0), 180)
    return c


@gf.cell(tags=["fixed", "laser"])
def laser_half_with_gc() -> gf.Component:
    """Half user cell with an integrated laser and a grating coupler feeding o1.

    Cornerstone Run 50 cell ``SOI220_Half_LD_DC1550HP01_Integrated_with_GC``.
    """
    c = _import_run50("SOI220_Half_LD_DC1550HP01_Integrated_with_GC")
    _add_optical_port(c, "o1", (-471.39, 1.57), 0)
    _add_fiber_port(c, "o2", (-645.541, 800), 180)
    return c
