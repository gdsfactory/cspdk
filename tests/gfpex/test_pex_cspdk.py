"""Consistency checks for the gfpex default PEX rules of the CORNERSTONE heater platforms."""

from __future__ import annotations

import importlib

import pytest

gfpex = pytest.importorskip("gfpex.gfpex")

PLATFORMS = ["si220", "si340", "si500", "sin200", "sin300"]


@pytest.fixture(params=PLATFORMS)
def platform(request: pytest.FixtureRequest) -> tuple:
    """The rules techfile and LAYER map of one heater platform."""
    rules = importlib.import_module(f"cspdk.{request.param}.pex.rules")
    tech = importlib.import_module(f"cspdk.{request.param}.tech")
    return rules.DEFAULT_PEX_RULES_CONFIG, tech.LAYER


def test_round_trip(platform: tuple) -> None:
    """The config is a gfpex Techfile that survives a JSON round trip."""
    config, _ = platform
    assert isinstance(config, gfpex.Techfile)
    assert gfpex.Techfile.model_validate_json(config.to_json()) == config


def test_names_resolve(platform: tuple) -> None:
    """Every stack, contact and parasitic name resolves to a declared layer."""
    config, _ = platform
    names = {layer.name for layer in config.layers}
    computed = {c.layer_info.name for c in config.lvs_computed_layers}
    for stack_layer in config.process_stack.layers:
        if (metal := stack_layer.metal_layer) is not None:
            assert stack_layer.name in names
            if (contact := metal.contact_above) is not None:
                assert contact.name in names | computed
                assert {contact.layer_below, contact.metal_above} <= names
    parasitics = config.process_parasitics
    used = {r.layer_name for r in parasitics.resistance.layers}
    used |= {c.layer_name for c in parasitics.capacitance.substrates}
    used |= {c.layer_name for c in parasitics.capacitance.sidewalls}
    assert used <= names


def test_gds_pairs_match_layer_map(platform: tuple) -> None:
    """Drawing GDS pairs equal the platform LAYER map."""
    config, layer_map = platform
    for layer in config.layers:
        expected = tuple(getattr(layer_map, layer.name))
        assert (layer.drw_gds_pair.layer, layer.drw_gds_pair.datatype) == expected


def test_metals_stack_upwards(platform: tuple) -> None:
    """Metals do not overlap in z and the direct contact has a z range."""
    config, _ = platform
    metals = [
        layer.metal_layer
        for layer in config.process_stack.layers
        if layer.metal_layer is not None
    ]
    for lower, upper in zip(metals, metals[1:]):
        assert upper.z >= lower.z + lower.thickness - 1e-9
    assert config.build_layer_z_lookup()["HEATER_PAD_CON"]


def test_pad_on_heater_is_one_resistive_network(platform: tuple) -> None:
    """The pad touches the filament with no via: gfpex must still join the two rasters."""
    import networkx as nx
    from shapely.geometry import Point, box

    config, layer_map = platform
    heater, pad = tuple(layer_map.HEATER), tuple(layer_map.PAD)
    bounds = (Point(0, 0), Point(30, 10))
    net = gfpex.rasterize_net_into_vector_net_raster(
        "n1",
        {heater: [box(0, 4, 30, 6)], pad: [box(0, 0, 10, 10), box(20, 0, 30, 10)]},
        0.5,
        bounds,
        gfpex.LazyLayerRasterizationPolicy(),
    )
    network = gfpex.resistance_extraction(config, net, None).ok_value
    assert nx.number_connected_components(network) == 1
