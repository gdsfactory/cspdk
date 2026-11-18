"""Smoke tests for the gflvs LVS rules in cspdk.lvs.rules (skipped without gflvs)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest

gflvs = pytest.importorskip("gflvs")

import gdstk  # noqa: E402
from gflvs.gflvs_refactor import (  # noqa: E402
    ElectricalDeviceExtractionPolicy,
    WaveguideDeviceExtractionPolicy,
)
from gflvs.gflvs_schema import LayerBooleanOperation, LvsRunMode  # noqa: E402
from result import Ok  # noqa: E402

from cspdk.lvs import rules  # noqa: E402


def test_default_config_builds() -> None:
    """The default config is a gflvs LvsConfig."""
    assert isinstance(rules.DEFAULT_LVS_CONFIG, gflvs.LvsConfig)


def test_connectivity_symmetric() -> None:
    """Every tracer connection goes both ways."""
    assert all(
        a in rules.LAYER_CONNECTIVITY[b]
        for a, bs in rules.LAYER_CONNECTIVITY.items()
        for b in bs
    )


@pytest.mark.parametrize("process", rules.PROCESSES)
def test_template_layers_in_pdk(process: str) -> None:
    """Every layer used by templates, tracer and fitter is a layer of the sub-PDK."""
    r = rules.process_rules(process)
    pdk_layers = {(layer.layer, layer.datatype) for layer in r.layer_map}
    used = (
        rules._leaf_layers(*r.heater_templates)
        | set(r.connectivity)
        | r.tracer_layers
        | r.waveguide_layers
    )
    assert used <= pdk_layers


@pytest.mark.parametrize("process", rules.PROCESSES)
def test_terminals_use_markers(process: str) -> None:
    """Every terminal op ANDs in a device marker layer."""
    r = rules.process_rules(process)
    for template in r.heater_templates:
        if template.device_name in rules.NO_MARKER_TEMPLATES:
            continue
        for terminal in template.device_terminal_template:
            assert _positive_leaves(terminal.layer_composed_op) & r.marker_layers, (
                template.device_name,
                terminal.terminal_name,
            )


def _positive_leaves(op: Any) -> set[tuple[int, int]]:
    """Leaf layers a region must lie on: the right operand of a NOT is excluded."""
    if op.WhichOneof("layer_left") == "layer_ref_left":
        left = {(op.layer_ref_left.layer.layer, op.layer_ref_left.layer.datatype)}
    else:
        left = _positive_leaves(op.layer_op_left)
    if op.bool_op == LayerBooleanOperation.LAYER_BOOLEAN_OPERATION_NOT:
        return left
    which = op.WhichOneof("layer_right")
    if which == "layer_ref_right":
        ref = op.layer_ref_right
        return left | {(ref.layer.layer, ref.layer.datatype)}
    return left | (_positive_leaves(op.layer_op_right) if which else set())


def test_unknown_process_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """A PROCESS outside PROCESSES is rejected."""
    monkeypatch.setattr(rules, "PROCESS", "si999")
    with pytest.raises(ValueError, match="si999"):
        rules.build_lvs_config()


def test_flags_toggle(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each flag switches its piece of the config."""
    default = rules.build_lvs_config()
    assert default.netlist_tracer_config.galvanic_drawing_layers
    assert default.netlist_tracer_config.waveguide_drawing_layers
    assert default.waveguide_device_extraction_config.pcell_generators
    assert default.run_mode == LvsRunMode.LVS_RUN_MODE_FLAT
    heater_names = {t.device_name for t in rules.HEATER_TEMPLATES}
    waveguide_names = {t.device_name for t in rules.WAVEGUIDE_TEMPLATES}
    assert {t.device_name for t in default.device_templates} == (
        heater_names | waveguide_names
    )

    with monkeypatch.context() as m:
        m.setattr(rules, "ENABLE_ELECTRICAL_TRACING", False)
        assert (
            not rules.build_lvs_config().netlist_tracer_config.galvanic_drawing_layers
        )

    with monkeypatch.context() as m:
        m.setattr(rules, "ENABLE_WAVEGUIDE_TRACING", False)
        assert (
            not rules.build_lvs_config().netlist_tracer_config.waveguide_drawing_layers
        )

    with monkeypatch.context() as m:
        m.setattr(rules, "ENABLE_DEVICE_EXTRACTION", False)
        cfg = rules.build_lvs_config()
        assert cfg.device_templates == []
        assert not cfg.waveguide_device_extraction_config.pcell_generators

    with monkeypatch.context() as m:
        m.setattr(rules, "ENABLE_PARAMETER_CHECK", False)
        cfg = rules.build_lvs_config()
        assert cfg.device_templates
        assert all(t.properties["blackbox"] == "true" for t in cfg.device_templates)
    assert all("blackbox" not in t.properties for t in rules.HEATER_TEMPLATES)

    with monkeypatch.context() as m:
        m.setattr(rules, "ENABLE_HEATER_TEMPLATES", False)
        names = {t.device_name for t in rules.build_lvs_config().device_templates}
        assert names == waveguide_names

    with monkeypatch.context() as m:
        m.setattr(rules, "ENABLE_WAVEGUIDE_DEVICE_FITTING", False)
        cfg = rules.build_lvs_config()
        assert {t.device_name for t in cfg.device_templates} == heater_names
        assert not cfg.waveguide_device_extraction_config.pcell_generators

    with monkeypatch.context() as m:
        m.setattr(rules, "HIERARCHICAL_RUN", True)
        assert rules.build_lvs_config().run_mode == LvsRunMode.LVS_RUN_MODE_HIERARCHICAL

    with monkeypatch.context() as m:
        m.setattr(rules, "PROCESS", "sin300")
        cfg = rules.build_lvs_config()
        assert cfg.waveguide_device_extraction_config.waveguide_layers == {
            (203, 0),
            (204, 0),
        }
    rules.build_lvs_config()  # re-activate the default sub-PDK


@pytest.mark.parametrize(
    "pcell, kwargs, device, n_terminals, params",
    [
        (
            "straight_heater_metal",
            {"length": 320},
            "straight_heater",
            3,
            {"length": 329.8},
        ),
        (
            "straight_heater_metal",
            {"length": 100},
            "straight_heater",
            3,
            {"length": 109.8},
        ),
        ("heater", {"length": 150}, "straight_heater", 2, {"length": 150.0}),
        ("spiral_racetrack_heater", {}, "straight_heater", 3, {}),
    ],
)
def test_pcell_extracts(
    pcell: str,
    kwargs: dict[str, Any],
    device: str,
    n_terminals: int,
    params: dict[str, float],
    tmp_path: Path,
) -> None:
    """A heater PCell extracts exactly one device with the verified terminals and parameters."""
    cfg = rules.build_lvs_config()  # si220_cband, activates the sub-PDK
    component = rules.process_rules(rules.PROCESS).pdk.get_component(pcell, **kwargs)
    gds = tmp_path / "cell.gds"
    component.write_gds(gds)
    top = next(c for c in gdstk.read_gds(str(gds)).cells if c.name == component.name)
    results = ElectricalDeviceExtractionPolicy().execute(
        top, rules.DEVICE_TEMPLATES, cfg.electrical_device_extraction_config
    )
    devices = [r.ok_value for r in results if isinstance(r, Ok)]
    assert len(devices) == 1, results
    (extracted,) = devices
    assert extracted.class_name == device
    assert len(extracted.terminals) == n_terminals
    values = {p.name: p.value for p in extracted.parameters}
    for name, expected in params.items():
        assert math.isclose(values[name], expected, rel_tol=1e-3), values


def test_heater_terminals_on_traced_nets(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Tracer plus extraction: the heater's two pins sit on two distinct PAD nets."""
    monkeypatch.setattr(rules, "ENABLE_WAVEGUIDE_DEVICE_FITTING", False)
    cfg = rules.build_lvs_config()
    component = rules.process_rules(rules.PROCESS).pdk.get_component(
        "straight_heater_metal"
    )
    gds = tmp_path / "heater.gds"
    component.write_gds(gds)
    _, l2n, _ = gflvs.layout_to_netlist(
        gdstk.read_gds(str(gds)),
        component.name,
        device_templates=cfg.device_templates,
        nettracer_config=cfg.netlist_tracer_config,
        electrical_device_extraction_config=cfg.electrical_device_extraction_config,
        waveguide_device_extraction_config=cfg.waveguide_device_extraction_config,
        run_mode=cfg.run_mode,
    )
    (heater,) = [d for d in l2n.devices if d.class_name == "straight_heater"]
    pins = [t for t in heater.terminals if t.terminal_type == rules.PIN]
    assert [t.name for t in pins] == ["l", "r"]  # the PCell's electrical pin names
    pin_nets = [t.parent_net.net_name for t in pins]
    assert len(pin_nets) == 2 and all(pin_nets), pin_nets
    assert len(set(pin_nets)) == 2, pin_nets
    # a floating terminal (leftmost layer not traced) gets a shapeless Net$ net
    traced = {net.name for net in l2n.nets if net.connected_shapes}
    assert set(pin_nets) <= traced, (pin_nets, traced)


@pytest.mark.parametrize("process", rules.PROCESSES)
def test_waveguide_bounds_build(process: str) -> None:
    """No generator raises at any bounds corner or the centre."""
    rules.check_bounds(rules.process_rules(process).waveguide_devices)


@pytest.mark.slow
@pytest.mark.parametrize(
    "process, device, true_params",
    [
        (
            "si220_cband",
            "mmi1x2",
            {
                "length_mmi": 27.3,
                "width_mmi": 5.0,
                "gap_mmi": 1.3,
                "length_taper": 12.0,
                "width_taper": 1.2,
            },
        ),
        (
            "si220_cband",
            "mmi2x2",
            {
                "length_mmi": 45.0,
                "width_mmi": 6.0,
                "gap_mmi": 0.6,
                "length_taper": 15.0,
                "width_taper": 1.4,
            },
        ),
        ("si220_cband", "coupler", {"length": 20.0, "gap": 0.3}),
        ("si220_cband", "coupler_rib", {"length": 25.0, "gap": 0.3}),
        (
            "si220_cband",
            "coupler_ring",
            {"length_x": 6.0, "gap": 0.22, "radius": 12.0, "length_extension": 5.0},
        ),
        (
            "si220_cband",
            "grating_coupler_elliptical",
            {"wavelength": 1.5, "grating_line_width": 0.3},
        ),
        (
            "si220_oband",
            "mmi1x2",
            {
                "length_mmi": 30.0,
                "width_mmi": 5.0,
                "gap_mmi": 1.2,
                "length_taper": 12.0,
                "width_taper": 1.2,
            },
        ),
        ("si340", "coupler", {"length": 20.0, "gap": 0.25, "n_variant": 2}),
        (
            "si340",
            "mmi1x2",
            {
                "width": 0.4,
                "length_mmi": 42.6,
                "width_mmi": 6.0,
                "gap_mmi": 1.51,
                "length_taper": 20.0,
                "width_taper": 1.5,
            },
        ),
        (
            "si340",
            "grating_coupler_elliptical",
            {"wavelength": 1.31, "grating_line_width": 0.35, "n_variant": 1},
        ),
        (
            "si500",
            "mmi2x2",
            {
                "width": 0.4,
                "length_mmi": 50.2,
                "width_mmi": 6.0,
                "gap_mmi": 0.4,
                "length_taper": 20.0,
                "width_taper": 1.6,
            },
        ),
        (
            "sin200",
            "mmi1x2",
            {
                "width": 0.27,
                "length_mmi": 21.6,
                "width_mmi": 3.4,
                "gap_mmi": 0.4,
                "length_taper": 40.0,
                "width_taper": 1.3,
            },
        ),
        ("sin200", "coupler", {"length": 20.0, "gap": 0.3, "n_variant": 1}),
        ("sin300", "coupler", {"length": 20.0, "gap": 0.25, "n_variant": 0}),
        (
            "sin300",
            "grating_coupler_elliptical",
            {"wavelength": 1.31, "grating_line_width": 0.482, "n_variant": 1},
        ),
    ],
)
def test_waveguide_fit(
    process: str, device: str, true_params: dict[str, float]
) -> None:
    """A cell generated with *true_params* is fitted back to them."""
    r = rules.process_rules(process)
    template, generator = next(
        d for d in r.waveguide_devices if d[0].device_name == device
    )
    cfg = rules.waveguide_config(r)
    cell = generator(**true_params)
    result = WaveguideDeviceExtractionPolicy().execute(
        cell, [t for t, _ in r.waveguide_devices], cfg
    )
    assert isinstance(result, Ok), result
    values = {p.name: p.value for p in result.ok_value.parameters}
    for name, expected in true_params.items():
        if name.startswith("n_"):  # gflvs reports the unsnapped index (B4), e.g. 1.879
            assert rules.snap(name, values[name]) == expected, values
            continue
        assert math.isclose(
            values[name], expected, rel_tol=rules.PHOTONIC_PARAMETER_RELATIVE_TOLERANCE
        ), values
