"""Private gfpex techfile builder shared by the CORNERSTONE heater platforms.

Every CORNERSTONE platform with metal (si220, si340, si500, sin200, sin300) has the same electrical stack: a Si
substrate, a thermal SiO2 BOX, the patterned core (Si or SiN, optical only), a SiO2 top cladding, TiN heater
filaments (GDS 39) lifted off on the cladding and heater contact pads (GDS 41) lifted off on the filament ends. There
is no via: the pads touch the filaments where they overlap, and nothing is deposited after the pads (air above).
Each platform's ``cspdk.<platform>.pex.rules`` passes its own sourced thicknesses to ``build_heater_pad_techfile``.

Stack convention (gf-pex / KPEX): z = 0 is the top of the conducting Si substrate, so the BOX is the field oxide
from z = 0 to its thickness, and the heater sits at z = BOX + core + cladding.

Provenance tags on every number:
    FOUNDRY <source>   value stated by a CORNERSTONE design guideline
    DERIVED <formula>  computed from foundry values (C = 8.854·k/t aF/µm², Rs = ρ/t, series Σt/k, ...)
    ASSUMED <source>   borrowed from the closest open PDK (gf-pex gfpexconfg/ihp_sg13g2.py:line), literature or a
                       cspdk placeholder
    PALACE-FIT         fitted with gf-pex benchmarks/palace_benchmark.py (pending: TODO(palace))

Units (gfpex techfile): µm; sheet resistance mΩ/sq; area and substrate capacitance aF/µm²; sidewall in Magic form,
``capacitance`` aF/µm at a 1 µm reference spacing, C = capacitance·L/(s + offset) (a parallel plate gives 8.854·k·t
with offset 0).
"""

from __future__ import annotations

from gfpex.gfpex import (
    CapacitanceInfo,
    ComputedLayerInfo,
    ComputedLayerKind,
    Contact,
    FieldOxideLayer,
    GDSPair,
    LayerInfo,
    LayerPurpose,
    LayerResistance,
    MetalLayer,
    ProcessParasiticsInfo,
    ProcessStackInfo,
    ResistanceInfo,
    SidewallCapacitance,
    SimpleDielectricLayer,
    StackLayerInfo,
    StackLayerType,
    SubstrateCapacitance,
    SubstrateLayer,
    Techfile,
)

VACUUM_PERMITTIVITY_AF_PER_UM = 8.854
OHM_M_TO_MILLIOHM_UM = 1e9  # ρ [Ω·m] / t [µm] → Rs [mΩ/sq]

BOX_K = 3.95  # ASSUMED gf-pex gfpexconfg/ihp_sg13g2.py:845 (field oxide); CORNERSTONE gives optical n only
CLADDING_K = 4.1  # ASSUMED gf-pex gfpexconfg/ihp_sg13g2.py:886 (SiO2 ILD); CORNERSTONE gives optical n only
AIR_K = 1.0  # ASSUMED gf-pex gfpexconfg/ihp_sg13g2.py:1153; nothing is deposited after the pad lift-off
SUBSTRATE_THICKNESS_UM = 0.28  # ASSUMED gf-pex gfpexconfg/ihp_sg13g2.py:791
SIDE_HALO_UM = 8.0  # ASSUMED gf-pex gfpexconfg/ihp_sg13g2.py:1162

# TODO(foundry): no CORNERSTONE heater or pad thickness / resistivity; these are literature films on the cspdk
# placeholder thicknesses
TIN_RESISTIVITY_OHM_M = 1.0e-6  # ASSUMED literature, sputtered TiN thin film ~100 µΩ·cm
AL_RESISTIVITY_OHM_M = 2.8e-8  # ASSUMED literature, thin-film Al 2.8 µΩ·cm (the Ti adhesion layer is ignored)

HEATER = "HEATER"
PAD = "PAD"
# the pads touch the filaments with no via: a computed contact on the pad drawing layer joins them (the
# HEATER ∩ PAD overlap). Naming the contact after a metal would overwrite that metal's z in build_layer_z_lookup
HEATER_PAD_CONTACT = "HEATER_PAD_CON"


def build_heater_pad_techfile(
    name: str,
    heater_gds: tuple[int, int],
    pad_gds: tuple[int, int],
    box_thickness_um: float,
    core_thickness_um: float,
    cladding_thickness_um: float,
    heater_thickness_um: float,
    pad_thickness_um: float,
    palace_fit: dict[str, dict[str, float]],
) -> Techfile:
    """Return the gfpex techfile of one CORNERSTONE heater platform from its sourced stack thicknesses.

    ``palace_fit`` maps HEATER / PAD to their Palace-fitted ``perimeter`` (aF/µm), ``sidewall`` (aF/µm at 1 µm) and
    ``sidewall_offset`` (µm), from doplaydo/pdk-gfpex-testdata ``cspdk/<platform>/outputs/fit.json``.
    """
    tech = Techfile(name=name)
    tech.layers = [
        LayerInfo(
            purpose=LayerPurpose.PURPOSE_METAL,
            name=HEATER,
            drw_gds_pair=GDSPair(layer=heater_gds[0], datatype=heater_gds[1]),
            description="TiN heater filaments, lift-off on the top cladding",
        ),
        LayerInfo(
            purpose=LayerPurpose.PURPOSE_METAL,
            name=PAD,
            drw_gds_pair=GDSPair(layer=pad_gds[0], datatype=pad_gds[1]),
            description="Ti/Al heater contact pads, lift-off on the filament ends",
        ),
    ]
    tech.lvs_computed_layers = [
        ComputedLayerInfo(
            kind=ComputedLayerKind.KIND_REGULAR,
            layer_info=LayerInfo(
                purpose=LayerPurpose.PURPOSE_CONTACT,
                name=HEATER_PAD_CONTACT,
                drw_gds_pair=GDSPair(layer=pad_gds[0], datatype=pad_gds[1]),
                description="Direct pad-on-filament contact (HEATER ∩ PAD), no via layer",
            ),
            original_layer_name=PAD,
        )
    ]

    # oxide under the heater over the etched field: the cladding fills the core height (planar cladding assumed)
    cladding_top_um = (
        box_thickness_um + core_thickness_um + cladding_thickness_um
    )  # DERIVED
    heater_z_um = cladding_top_um
    pad_z_um = (
        heater_z_um + heater_thickness_um
    )  # DERIVED: the pad lands on the filament
    tech.process_stack = ProcessStackInfo(
        layers=[
            StackLayerInfo(
                name="subs",
                layer_type=StackLayerType.LAYER_TYPE_SUBSTRATE,
                substrate_layer=SubstrateLayer(
                    height=0.0, thickness=SUBSTRATE_THICKNESS_UM, reference="box"
                ),
            ),
            StackLayerInfo(
                name="box",
                layer_type=StackLayerType.LAYER_TYPE_FIELD_OXIDE,
                field_oxide_layer=FieldOxideLayer(dielectric_k=BOX_K),
            ),
            StackLayerInfo(
                name="cladding",
                layer_type=StackLayerType.LAYER_TYPE_SIMPLE_DIELECTRIC,
                simple_dielectric_layer=SimpleDielectricLayer(
                    dielectric_k=CLADDING_K, reference="box"
                ),
            ),
            StackLayerInfo(
                name=HEATER,
                layer_type=StackLayerType.LAYER_TYPE_METAL,
                metal_layer=MetalLayer(
                    z=heater_z_um,
                    thickness=heater_thickness_um,
                    contact_above=Contact(
                        name=HEATER_PAD_CONTACT,
                        layer_below=HEATER,
                        metal_above=PAD,
                        thickness=0.0,
                    ),  # TODO(palace): width/spacing/border 0, a pad-on-filament overlap has no cut geometry
                ),
            ),
            StackLayerInfo(
                name=PAD,
                layer_type=StackLayerType.LAYER_TYPE_METAL,
                metal_layer=MetalLayer(z=pad_z_um, thickness=pad_thickness_um),
            ),
            StackLayerInfo(
                name="air",
                layer_type=StackLayerType.LAYER_TYPE_SIMPLE_DIELECTRIC,
                simple_dielectric_layer=SimpleDielectricLayer(
                    dielectric_k=AIR_K, reference="cladding"
                ),
            ),
        ]
    )

    # area capacitance to the substrate through the series BOX + cladding, oxide-only path over the etched field
    oxide_t_over_k_um = (
        box_thickness_um / BOX_K
        + (core_thickness_um + cladding_thickness_um) / CLADDING_K
    )
    substrate_area_af_um2 = (
        VACUUM_PERMITTIVITY_AF_PER_UM / oxide_t_over_k_um
    )  # DERIVED 8.854/Σ(t/k)
    tech.process_parasitics = ProcessParasiticsInfo(
        side_halo=SIDE_HALO_UM,
        resistance=ResistanceInfo(
            layers=[
                # DERIVED Rs = ρ/t
                LayerResistance(
                    layer_name=HEATER,
                    resistance=TIN_RESISTIVITY_OHM_M
                    / heater_thickness_um
                    * OHM_M_TO_MILLIOHM_UM,
                ),
                LayerResistance(
                    layer_name=PAD,
                    resistance=AL_RESISTIVITY_OHM_M
                    / pad_thickness_um
                    * OHM_M_TO_MILLIOHM_UM,
                ),
            ],
            # TODO(foundry): pad-on-filament interface resistance NOT FOUND; no contact or via entry
        ),
        capacitance=CapacitanceInfo(
            # the pad mostly sits on the cladding, so it shares the heater's oxide path (the 0.15 µm filament step is
            # ignored); area DERIVED, perimeter (fringe) PALACE-FIT
            substrates=[
                SubstrateCapacitance(
                    layer_name=metal,
                    area_capacitance=substrate_area_af_um2,
                    perimeter_capacitance=palace_fit[metal]["perimeter"],
                )
                for metal in (HEATER, PAD)
            ],
            # no overlap or side-overlap entry: a pad over a filament is the direct contact, both on one net
            # PALACE-FIT: the coupling at the ≥ 5-10 µm DRC spacings decays faster than 1/s (the substrate is closer
            # than the spacing), so the fit sits on the two closest spacings with a negative offset
            sidewalls=[
                SidewallCapacitance(
                    layer_name=metal,
                    capacitance=palace_fit[metal]["sidewall"],
                    offset=palace_fit[metal]["sidewall_offset"],
                )
                for metal in (HEATER, PAD)
            ],
        ),
    )
    return tech
