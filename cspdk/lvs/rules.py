"""gflvs LVS rules for the Cornerstone PDK (cspdk), all sub-PDKs.

Usage::

    import gflvs
    from cspdk.lvs import rules

    gflvs.lvs(netlist, lib, rules.DEFAULT_LVS_CONFIG)  # si220, C-band

    rules.PROCESS = "sin300"  # any of rules.PROCESSES
    gflvs.lvs(netlist, lib, rules.build_lvs_config())

``build_lvs_config()`` reads the flags below at call time. Building it has no
side effects: the waveguide generators activate the sub-PDK picked by
``PROCESS`` on every call, as the cspdk sub-PDKs share cell names.

Targets the gflvs ``dev`` API (``gflvs.gflvs_refactor``: ``LvsConfig``,
``NetlistTracerConfig``, ``ElectricalDeviceExtractionConfig``,
``WaveguideDeviceExtractionConfig``; protos from ``gflvs.gflvs_schema``).
gflvs is not on PyPI: install it from its repository to use this module.

gflvs extracts electrical devices from a cell's own polygons only (depth=0), so
device PCells must be flat; ``spiral_racetrack_heater`` is flattened for this.
"""

from __future__ import annotations

import importlib
import itertools
from collections.abc import Callable, Iterable, Sequence
from functools import cache, partial
from typing import Any, NamedTuple

import gdstk
import numpy as np
from gdsfactory.pdk import Pdk
from gflvs import (
    ElectricalDeviceExtractionConfig,
    LvsConfig,
    NetlistTracerConfig,
    WaveguideDeviceExtractionConfig,
)
from gflvs.gflvs_refactor import HashingOptimizationConfig
from gflvs.gflvs_schema import (
    DeviceParameterTemplate,
    DeviceTemplate,
    DeviceTerminalTemplate,
    DeviceTerminalType,
    LayerGds,
    LayerOffsetOperation,
    LayerOperation,
    LayerRef,
    LvsRunMode,
)
from pymoo.algorithms.soo.nonconvex.de import DE  # type: ignore[import-untyped]
from pymoo.termination import get_termination  # type: ignore[import-untyped]

# ── Flags ───────────────────────────────────────────────────────────
# si220_cband | si220_oband | si340 | si500 | sin200 | sin300 | si_sus | ge_on_si | si220_ebeam_event
PROCESS = "si220_cband"
ENABLE_ELECTRICAL_TRACING = True  # trace PAD nets; False → tracer galvanic layers empty
ENABLE_WAVEGUIDE_TRACING = (
    True  # trace waveguide nets; False → waveguide_drawing_layers empty
)
ENABLE_DEVICE_EXTRACTION = True  # use the templates; False → connectivity-only LVS
ENABLE_HEATER_TEMPLATES = (
    True  # heater (straight_heater) template; False → heaters not extracted
)
ENABLE_PARAMETER_CHECK = True  # False → every template blackboxed (topology only)
ENABLE_WAVEGUIDE_DEVICE_FITTING = (
    True  # fit photonic pcells via PCELL_GENERATORS; False → not fitted
)
SKIP_COARSE_SEARCH = (
    True  # fitter: skip the coarse global search (30 s vs 1325 s on a GC)
)
HIERARCHICAL_RUN = False  # True → LVS_RUN_MODE_HIERARCHICAL, else FLAT
VERBOSE = False  # True → LvsConfig.verbose=1
# ponytail: loosened until gflvs snaps fitted values (gflvs B4: 1 nm on a 0.2 um gap is 5e-3)
PHOTONIC_PARAMETER_RELATIVE_TOLERANCE = 5e-3
REFINEMENT_EVALUATIONS = (
    6000  # fitter XOR-search budget; 6-parameter MMIs need about 6000
)

PROCESSES: tuple[str, ...] = (
    "si220_cband",
    "si220_oband",
    "si340",
    "si500",
    "sin200",
    "sin300",
    "si_sus",
    "ge_on_si",
    "si220_ebeam_event",
)

# ── Helpers ─────────────────────────────────────────────────────────
# ponytail: duplicated per PDK repo; move to gflvs when it ships helpers
LayerKey = tuple[int, int]
PIN = DeviceTerminalType.DEVICE_TERMINAL_TYPE_PIN
PORT = DeviceTerminalType.DEVICE_TERMINAL_TYPE_PORT
DBU = 1e-3  # um


def L(layer: Any) -> LayerRef:
    """LayerRef of a PDK LayerMap entry."""
    return LayerRef(layer=LayerGds(layer=layer.layer, datatype=layer.datatype))


def op(x: LayerRef | LayerOperation) -> LayerOperation:
    """Wrap a bare layer as a LayerOperation."""
    return x if isinstance(x, LayerOperation) else LayerOperation(layer_ref_left=x)


def grow(x: LayerRef | LayerOperation, d: float) -> LayerOperation:
    """Isotropic sizing of *x* by *d* um."""
    side = "layer_ref_left" if isinstance(x, LayerRef) else "layer_op_left"
    return LayerOperation(
        **{side: x}, offset_op=LayerOffsetOperation(offset_x=d, offset_y=d)
    )


def pin(name: str, x: LayerRef | LayerOperation) -> DeviceTerminalTemplate:
    """Electrical terminal; the leftmost layer is the routing metal."""
    return DeviceTerminalTemplate(
        terminal_name=name, terminal_type=PIN, layer_composed_op=op(x)
    )


def port(name: str, x: LayerRef | LayerOperation) -> DeviceTerminalTemplate:
    """Optical terminal; the leftmost layer is the waveguide layer."""
    return DeviceTerminalTemplate(
        terminal_name=name, terminal_type=PORT, layer_composed_op=op(x)
    )


def par(
    name: str, x: LayerRef | LayerOperation, extractor: str = ""
) -> DeviceParameterTemplate:
    """Parameter region; *extractor* is a built-in gflvs extractor name."""
    p = DeviceParameterTemplate(name=name, layer_composed_op=op(x))
    if extractor:
        p.properties["extractor"] = extractor
    return p


def tpl(
    name: str,
    cls: str,
    terms: Sequence[DeviceTerminalTemplate],
    params: Sequence[DeviceParameterTemplate] = (),
) -> DeviceTemplate:
    """Electrical device template."""
    return DeviceTemplate(
        device_name=name,
        device_class_name=cls,
        device_terminal_template=list(terms),
        device_parameter_template=list(params),
    )


def bbox_long(polygons: list[gdstk.Polygon]) -> float:
    """Long side of the region's bounding box; nan when empty."""
    if not polygons:
        return float("nan")
    points = np.vstack([p.points for p in polygons])
    return float(max(points.max(axis=0) - points.min(axis=0)))


def _leaf_layers(*ops: Any) -> set[LayerKey]:
    """Every (layer, datatype) read by the ops, templates or terminals given."""
    out: set[LayerKey] = set()
    for x in ops:
        if isinstance(x, DeviceTemplate):
            out |= _leaf_layers(
                *x.device_terminal_template, *x.device_parameter_template
            )
        elif isinstance(x, DeviceTerminalTemplate | DeviceParameterTemplate):
            out |= _leaf_layers(x.layer_composed_op)
        elif isinstance(x, LayerRef):
            out.add((x.layer.layer, x.layer.datatype))
        elif isinstance(x, LayerOperation):
            for side in ("left", "right"):
                which = x.WhichOneof(f"layer_{side}")
                if which:
                    out |= _leaf_layers(getattr(x, which))
    return out


# ── Waveguide declarator (waveguide-contract.md §2) ──────────────────
WaveguideDevice = tuple[DeviceTemplate, Callable[..., gdstk.Cell]]


def snap(name: str, value: float) -> float | int:
    """Counts (n_*) to int, port widths (width*) to 2 dbu, the rest to 1 dbu.

    kfactory rejects odd-dbu port widths.
    """
    if name.startswith("n_"):
        return round(value)
    grid = 2 * DBU if name.startswith("width") else DBU
    return round(round(value / grid) * grid, 6)


def component_to_gdstk_cell(component: Any) -> gdstk.Cell:
    """Flat gdstk copy of a gdsfactory component."""
    cell = gdstk.Cell(str(component.name))
    for (layer, datatype), polygons in component.get_polygons_points(
        by="tuple"
    ).items():
        cell.add(
            *(gdstk.Polygon(points.tolist(), layer, datatype) for points in polygons)
        )
    return cell


def waveguide_device(
    class_name: str,
    *pcells: Callable[..., Any],
    pdk: Pdk,
    **bounds: tuple[float, float],
) -> WaveguideDevice:
    """Template and generator of one fitted pcell, declared together.

    Several *pcells* are variants whose cells share a name prefix (one per
    cross-section, e.g. ``coupler_sc``/``coupler_so``/``coupler_rc``); the
    fitted ``n_variant`` index picks among them. *pdk* is activated before every
    call, as the sub-PDKs share cell names.
    """
    if len(pcells) > 1:
        bounds["n_variant"] = (0, len(pcells) - 1)
    template = DeviceTemplate(
        device_name=class_name,
        device_class_name=class_name,
        device_parameter_template=[
            DeviceParameterTemplate(name=n, param_bound_min=lo, param_bound_max=hi)
            for n, (lo, hi) in bounds.items()
        ],
    )

    def generator(**parameters: float) -> gdstk.Cell:
        kwargs = {n: snap(n, v) for n, v in parameters.items()}
        pcell = pcells[int(kwargs.pop("n_variant", 0))]
        pdk.activate()
        return component_to_gdstk_cell(pcell(**kwargs))

    return template, generator


def check_bounds(devices: Iterable[WaveguideDevice]) -> None:
    """Build every bounds corner and the centre: a raising generator aborts LVS."""
    for template, generator in devices:
        names = [p.name for p in template.device_parameter_template]
        box = [
            (p.param_bound_min, p.param_bound_max)
            for p in template.device_parameter_template
        ]
        # every integer of an n_* range (each variant), the two ends of the others
        axes = [
            range(int(lo), int(hi) + 1) if name.startswith("n_") else (lo, hi)
            for name, (lo, hi) in zip(names, box)
        ]
        for point in [
            *itertools.product(*axes),
            tuple((lo + hi) / 2 for lo, hi in box),
        ]:
            generator(**dict(zip(names, point)))


# ── Per-process rules ───────────────────────────────────────────────
# ponytail: tuned on PCell geometry; re-measure if PCell rules change.
# Farthest HEATER - PAD point from the waveguide: 10.0 um (spiral_racetrack_heater),
# 5.0 um (straight_heater_metal, both bands, with/without undercut), 3.0 um (meander).
HEATER_WG_REACH = 15.0
# Bounds: waveguide-cspdk.md, widened to cover every registered variant of
# the current sub-PDK (PDK defaults x ~0.5-2). Lower bounds are then raised to
# the sub-PDK's design-rule minimums, below which the cells raise (_RULE_MINIMUMS).
COUPLER_BOUNDS = dict(length=(1.0, 50.0), gap=(0.15, 0.6))
GC_ELLIPTICAL_BOUNDS = dict(wavelength=(1.2, 1.7), grating_line_width=(0.1, 0.6))
SI_MMI_BOUNDS = dict(
    length_mmi=(10.0, 80.0),
    width_mmi=(3.0, 12.0),
    gap_mmi=(0.2, 3.0),
    length_taper=(5.0, 40.0),
    width_taper=(0.8, 3.0),
)
# fitted parameter -> TECH attribute holding its cspdk._checks.check_min minimum
_RULE_MINIMUMS = {
    "gap": "gap_min",
    "gap_mmi": "gap_min",
    "grating_line_width": "grating_gap_min",
}


class ProcessRules(NamedTuple):
    """Everything PROCESS selects."""

    pdk: Pdk
    layer_map: Any
    heater_templates: list[DeviceTemplate]
    marker_layers: frozenset[LayerKey]
    connectivity: dict[LayerKey, list[LayerKey]]
    tracer_layers: frozenset[LayerKey]
    waveguide_tracing_layers: frozenset[LayerKey]
    waveguide_layers: frozenset[LayerKey]
    waveguide_devices: list[WaveguideDevice]


def _key(layer: Any) -> LayerKey:
    return int(layer.layer), int(layer.datatype)


# the unified si220 sub-PDK serves both bands; PROCESS picks the cross-sections
_SI220_BANDS = {"si220_cband": ("si220", "cband"), "si220_oband": ("si220", "oband")}


@cache
def process_rules(process: str) -> ProcessRules:
    """Layers, templates and generators of one PROCESS (sub-PDK, band)."""
    if process not in PROCESSES:
        raise ValueError(f"PROCESS={process!r}; valid: {', '.join(PROCESSES)}")
    module_name, band = _SI220_BANDS.get(process, (process, ""))
    module = importlib.import_module(f"cspdk.{module_name}")
    pdk: Pdk = module.get_pdk()
    LAYER = module.LAYER
    cells = module.cells

    def wd(
        class_name: str,
        *pcells: Callable[..., Any],
        **bounds: tuple[float, float],
    ) -> WaveguideDevice:
        for name, attribute in _RULE_MINIMUMS.items():
            if name in bounds:
                lo, hi = bounds[name]
                bounds[name] = (max(lo, getattr(module.tech.TECH, attribute)), hi)
        return waveguide_device(class_name, *pcells, pdk=pdk, **bounds)

    core = (
        LAYER.NITRIDE if hasattr(LAYER, "NITRIDE") else LAYER.WG
    )  # cross-section core
    devices: list[WaveguideDevice] = []
    if module_name == "si220":
        strip, rib = f"strip_{band}", f"rib_{band}"
        devices = [
            # *_rib MMIs differ only on SLAB, which is not compared: one template each
            # CONFIDENCE: HIGH on C-band (all six fitted back exactly, test_waveguide_fit);
            # on O-band HIGH for mmi1x2, MEDIUM for the rest (all corners build, not fitted)
            # TODO: a si220 layout mixing C- and O-band cells misfits under one PROCESS
            wd("mmi1x2", partial(cells.mmi1x2, cross_section=strip), **SI_MMI_BOUNDS),
            wd("mmi2x2", partial(cells.mmi2x2, cross_section=strip), **SI_MMI_BOUNDS),
            wd(
                "coupler", partial(cells.coupler, cross_section=strip), **COUPLER_BOUNDS
            ),
            wd(
                "coupler_rib",
                partial(cells.coupler_rib, cross_section=rib),
                **COUPLER_BOUNDS,
            ),
            wd(
                "coupler_ring",
                partial(cells.coupler_ring, cross_section=strip),
                length_x=(0.0, 20.0),
                gap=(0.15, 0.6),
                radius=(5.0, 50.0),
                length_extension=(0.0, 20.0),
            ),
            wd(
                "grating_coupler_elliptical",
                partial(cells.grating_coupler_elliptical, cross_section=strip),
                **GC_ELLIPTICAL_BOUNDS,
            ),
            # grating_coupler_rectangular: passthrough, see PASSTHROUGH_DEVICE_CLASS_NAMES
        ]
    elif module_name in ("si340", "si500", "sin200", "sin300"):
        suffixes = {
            "si340": ("sc", "so", "rc"),
            "si500": ("rc", "ro"),
            "sin200": ("n780", "n638", "n520"),
            "sin300": ("nc", "no"),
        }[module_name]

        def variants(name: str) -> list[Callable[..., Any]]:
            return [pdk.cells[f"{name}_{s}"] for s in suffixes]

        mmi_width = {
            "si340": (0.35, 0.85),
            "si500": (0.35, 0.5),
            "sin200": (0.25, 0.55),
            "sin300": (0.9, 1.25),
        }[module_name]
        mmi1x2_bounds, mmi2x2_bounds = {
            "sin200": (
                SI_MMI_BOUNDS | dict(length_taper=(5.0, 80.0)),
                SI_MMI_BOUNDS
                | dict(length_mmi=(40.0, 180.0), length_taper=(5.0, 80.0)),
            ),
            "sin300": (
                dict(
                    length_mmi=(20.0, 130.0),
                    width_mmi=(4.0, 24.0),
                    gap_mmi=(0.2, 3.0),
                    length_taper=(10.0, 100.0),
                    width_taper=(1.5, 8.0),
                ),
                dict(
                    length_mmi=(60.0, 400.0),
                    width_mmi=(6.0, 36.0),
                    gap_mmi=(0.2, 3.0),
                    length_taper=(10.0, 100.0),
                    width_taper=(1.5, 8.0),
                ),
            ),
        }.get(module_name, (SI_MMI_BOUNDS, SI_MMI_BOUNDS))
        devices = [
            # CONFIDENCE: HIGH where test_waveguide_fit covers it (si340 mmi1x2, coupler,
            # GC; si500 mmi2x2; sin200 mmi1x2, coupler; sin300 coupler, GC); MEDIUM for
            # the rest: every bounds corner builds, but no fit was run.
            # MMIs take width, so one pcell covers every cross-section
            wd("mmi1x2", cells.mmi1x2, width=mmi_width, **mmi1x2_bounds),
            wd("mmi2x2", cells.mmi2x2, width=mmi_width, **mmi2x2_bounds),
            # couplers fit gap and length only: fitting dx/dy raises min_bend_radius at corners
            # si500 coupler_ro raises min_bend_radius < 25 below gap ~0.231: start at its 0.236 default
            wd(
                "coupler",
                *variants("coupler"),
                **(
                    COUPLER_BOUNDS | dict(gap=(0.236, 0.6))
                    if module_name == "si500"
                    else COUPLER_BOUNDS
                ),
            ),
            wd(
                "grating_coupler_elliptical",
                *variants("grating_coupler_elliptical"),
                **(
                    GC_ELLIPTICAL_BOUNDS | dict(wavelength=(0.45, 0.9))
                    if module_name == "sin200"
                    else GC_ELLIPTICAL_BOUNDS
                ),
            ),
            # grating_coupler_rectangular: passthrough, see PASSTHROUGH_DEVICE_CLASS_NAMES
        ]
    # si_sus: its only device, grating_coupler_rectangular, is now a fixed foundry
    # GDS with no parameters to fit; ge_on_si: passthrough cells only;
    # si220_ebeam_event: event-only PDK, not covered by the swarm reports.

    heater_templates: list[DeviceTemplate] = []
    connectivity: dict[LayerKey, list[LayerKey]] = {}
    if hasattr(LAYER, "HEATER"):
        WG, HEATER, PAD = L(core), L(LAYER.HEATER), L(LAYER.PAD)
        # CONFIDENCE: HIGH for straight_heater_metal (L=320 → length 320) and heater()
        # (length=150 → 150, no waveguide: the filament span); MEDIUM for the meander/spiral.
        # TODO: straight_heater_meander/spiral_racetrack_heater length is a bounding box
        # (166.358 / 156.302), not the length kwarg; needs a path-length extractor
        heater_templates = [
            tpl(
                "heater",
                "straight_heater",
                # PAD leftmost: the terminal lands on the traced PAD net (rule 2).
                # "l/r" are the electrical pins of straight_heater_metal/_meander (their
                # ports l_e*/r_e* vary with band); left pad first, as gflvs orders pieces
                # along x. heater() calls them e1/e2, the flat spiral top/bot.
                # "o": one WG & HEATER piece spans the PCell's o1..o2, so no port name fits.
                [pin("l/r", PAD & HEATER), port("o", WG & HEATER)],
                [
                    # heater over the waveguide; a waveguide-less heater() falls back to
                    # its filament (HEATER - PAD beyond HEATER_WG_REACH of any WG)
                    par(
                        "length", HEATER & WG | HEATER - PAD - grow(WG, HEATER_WG_REACH)
                    ),
                    # filament between the pads: glues l, o and r into one device;
                    # gflvs compares only parameters the reference netlist names
                    par("span", HEATER - PAD),
                ],
            )
        ]
        # PAD only: the PDK's ("HEATER", "HEATER", "PAD") would short every heater pad to pad
        # TODO(issue: pending): tech.CONNECTIVITY ("HEATER", "HEATER", "PAD") shorts every heater
        connectivity = {_key(LAYER.PAD): []}

    waveguide_layers = {_key(core)}
    for extra in (
        "GRA",
        "NITRIDE_ETCH",
        "WG_MARK",
    ):  # grating/etch layer, never the slab
        if hasattr(LAYER, extra):
            waveguide_layers.add(_key(getattr(LAYER, extra)))
    tracing_layer = LAYER.WG_MARK if module_name == "si_sus" else core  # the xs layer
    return ProcessRules(
        pdk=pdk,
        layer_map=LAYER,
        heater_templates=heater_templates,
        marker_layers=frozenset({_key(LAYER.HEATER)} if heater_templates else ()),
        connectivity=connectivity,
        tracer_layers=frozenset(connectivity),
        waveguide_tracing_layers=frozenset({_key(tracing_layer)}),
        waveguide_layers=frozenset(waveguide_layers),
        waveguide_devices=devices,
    )


# ── Default-PROCESS exports ─────────────────────────────────────────
_DEFAULT = process_rules(PROCESS)
LAYER = _DEFAULT.layer_map
LAYER_CONNECTIVITY = _DEFAULT.connectivity
HEATER_TEMPLATES = _DEFAULT.heater_templates
DEVICE_TEMPLATES = HEATER_TEMPLATES  # electrical templates
MARKER_LAYERS = _DEFAULT.marker_layers
NO_MARKER_TEMPLATES: frozenset[str] = frozenset()  # the heater uses HEATER
PARAMETER_EXTRACTORS: dict[tuple[str, str], Callable[[list[gdstk.Polygon]], float]] = {
    ("heater", "length"): bbox_long,
    ("heater", "span"): bbox_long,
}
WAVEGUIDE_DEVICES = _DEFAULT.waveguide_devices
WAVEGUIDE_TEMPLATES = [t for t, _ in WAVEGUIDE_DEVICES]
PCELL_GENERATORS = {t.device_class_name: g for t, g in WAVEGUIDE_DEVICES}
# matched on cell_name.lower().startswith; "straight_l", not "straight", keeps straight_heater_*
PASSTHROUGH_DEVICE_CLASS_NAMES: tuple[str, ...] = (
    "via",
    "straight_l",
    "straight_rib",
    "straight_strip",
    "straight_metal",
    "bend",
    "taper",
    "wire",
    "pad",  # PAD is also the routing metal, so pads can't be templated
    "compass",
    "rectangle",
    # TODO(gflvs B5): fit grating_coupler_rectangular once the fitter lands integer
    # parameters; it fails today (n_periods 38.7-53.6 vs 40, 0.54-8.8% XOR), and a
    # failed fit is an extraction error even when blackboxed. Bounds to restore
    # (all corners build): period=(0.4, 1), n_periods=(10, 80), length_taper=(50, 500),
    # plus fill_factor=(0.3, 0.7) on si340/si500/sin200, period=(0.3, 1) and
    # width_grating=(3, 12) on sin200, period=(0.6, 2) on sin300; variants per sub-PDK.
    # No other cell name starts with it (si220's *_rib variant is the same device).
    # si_sus's grating_coupler_rectangular is a fixed foundry GDS with nothing to fit.
    "grating_coupler_rectangular",
)


def waveguide_config(rules: ProcessRules) -> WaveguideDeviceExtractionConfig:
    """Fitter config for one process, from the flags."""
    fitting = ENABLE_DEVICE_EXTRACTION and ENABLE_WAVEGUIDE_DEVICE_FITTING
    return WaveguideDeviceExtractionConfig(
        pcell_generators={t.device_class_name: g for t, g in rules.waveguide_devices}
        if fitting
        else {},
        waveguide_layers=rules.waveguide_layers,
        skip_coarse_search=SKIP_COARSE_SEARCH,
        refinement_optimization_config=HashingOptimizationConfig(
            DE(pop_size=20), get_termination("n_eval", REFINEMENT_EVALUATIONS)
        ),
        rotation_bounds_degrees=(0.0, 0.0),  # pcells are fitted in their own frame
        parameter_decimal_places=3,
    )


def _blackboxed(template: DeviceTemplate) -> DeviceTemplate:
    copy = DeviceTemplate()
    copy.CopyFrom(template)
    copy.properties["blackbox"] = "true"
    return copy


def build_lvs_config() -> LvsConfig:
    """LvsConfig for PROCESS from the module flags, read at call time."""
    rules = process_rules(PROCESS)
    templates: list[DeviceTemplate] = []
    if ENABLE_DEVICE_EXTRACTION:
        if ENABLE_HEATER_TEMPLATES:
            templates += rules.heater_templates
        if ENABLE_WAVEGUIDE_DEVICE_FITTING:
            templates += [t for t, _ in rules.waveguide_devices]
    if not ENABLE_PARAMETER_CHECK:
        templates = [_blackboxed(t) for t in templates]
    electrical = [t for t in templates if t.device_terminal_template]
    return LvsConfig(
        device_templates=templates,
        netlist_tracer_config=NetlistTracerConfig(
            layer_connectivity=rules.connectivity,
            galvanic_drawing_layers=rules.tracer_layers
            if ENABLE_ELECTRICAL_TRACING
            else frozenset(),
            waveguide_drawing_layers=rules.waveguide_tracing_layers
            if ENABLE_WAVEGUIDE_TRACING
            else frozenset(),
            passthrough_device_class_names=PASSTHROUGH_DEVICE_CLASS_NAMES,
        ),
        electrical_device_extraction_config=ElectricalDeviceExtractionConfig(
            galvanic_drawing_layers=frozenset(_leaf_layers(*electrical)),
            parameter_extractors=PARAMETER_EXTRACTORS,
        ),
        waveguide_device_extraction_config=waveguide_config(rules),
        parameter_relative_tolerance=PHOTONIC_PARAMETER_RELATIVE_TOLERANCE,
        run_mode=LvsRunMode.LVS_RUN_MODE_HIERARCHICAL
        if HIERARCHICAL_RUN
        else LvsRunMode.LVS_RUN_MODE_FLAT,
        verbose=int(VERBOSE),
    )


DEFAULT_LVS_CONFIG = build_lvs_config()
