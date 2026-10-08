"""Test routing with layer_transitions and auto-taper."""

import pytest


@pytest.mark.parametrize(
    "pdk_module,sample_module",
    [
        ("cspdk.si220", "cspdk.si220.samples.sample_routing"),
        ("cspdk.si500", "cspdk.si500.samples.sample_routing"),
        ("cspdk.sin300", "cspdk.sin300.samples.sample_routing"),
    ],
)
def test_sample_routing_different_widths(pdk_module: str, sample_module: str) -> None:
    """Test that routing two straights with different widths works."""
    import importlib

    pdk = importlib.import_module(pdk_module)
    pdk.PDK.activate()
    sample = importlib.import_module(sample_module)
    c = sample.sample_routing_different_widths()
    # one taper at each end, from each straight's width to the route's width
    tapers = [inst for inst in c.insts if inst.cell.name.startswith("taper")]
    assert len(tapers) == 2, (
        f"Expected 2 tapers, got {[inst.cell.name for inst in c.insts]}"
    )
