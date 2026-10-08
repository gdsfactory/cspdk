"""Generate the si220 coupler tables (cspdk.si220.models) with coupled-mode theory.

1. Mode-solve the even and odd supermodes of two coupled waveguides over a gap sweep.
2. Fit the index splitting to dn = A * exp(-gap / L), with log A and 1/L linear in wl.
3. Integrate kappa = pi * dn / wl along the gap profile of the coupler.

Raw splittings are cached in coupler_cmt_splitting.json. Not foundry validated;
agrees with 3D FDTD to within ~10% for a strip directional coupler.

Usage:
    python coupler_cmt.py [BAND KIND ...]   # e.g. "oband strip oband rib"
    python coupler_cmt.py --from-cache      # rebuild every table from cached splittings
"""

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import xarray as xr

HERE = Path(__file__).resolve().parent
MODELS = HERE.parent / "models"
CACHE = HERE / "coupler_cmt_splitting.json"
SLAB = 0.1
CONFIGS = {
    ("cband", "strip"): {"wavelengths": (1.5, 1.55, 1.6), "width": 0.45},
    ("cband", "rib"): {"wavelengths": (1.5, 1.55, 1.6), "width": 0.45},
    ("oband", "strip"): {"wavelengths": (1.26, 1.31, 1.36), "width": 0.40},
    ("oband", "rib"): {"wavelengths": (1.26, 1.31, 1.36), "width": 0.40},
}
GAPS = (0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5)
# table grids (wavelength axis: 21 points spanning the band)
GRIDS = {
    "coupler_racetrack": {
        "gap": np.linspace(0.05, 1.05, 21),
        "radius": np.linspace(5, 245, 13),
        "length_x": np.linspace(0, 28, 15),
    },
    "directional_coupler": {
        "gap": np.linspace(0.05, 0.75, 15),
        "radius": np.linspace(5, 85, 5),
        "length_x": np.linspace(0, 96, 25),
        "v_offset": np.linspace(5, 85, 5),
    },
}


def _splitting(task: tuple[str, float, float, float]) -> float:
    import gplugins.tidy3d.modes as m

    kind, wl, width, gap = task
    coupler = m.WaveguideCoupler(
        wavelength=wl,
        core_width=(width, width),
        core_thickness=0.22,
        slab_thickness=SLAB if kind == "rib" else 0.0,
        core_material="Si",
        clad_material="SiO2",
        gap=gap,
        num_modes=2,
        grid_resolution=40,
    )
    solver = coupler.waveguide.mode_solver
    n_eff = []
    for symmetry in (1, -1):  # even/odd about the gap centre: no spurious splitting
        data = solver.updated_copy(
            simulation=solver.simulation.updated_copy(symmetry=(symmetry, 0, 0))
        ).solve()
        n = np.real(np.asarray(data.n_eff).squeeze())
        te = np.asarray(data.pol_fraction.te).squeeze()
        n_eff.append(float(np.max(n[te > 0.5])))
    return abs(n_eff[0] - n_eff[1])


def solve_splitting(band: str, kind: str, workers: int = 6) -> dict:
    """Return the raw supermode index splitting {"gaps": [...], "dn": {wl: [...]}}."""
    cfg = CONFIGS[band, kind]
    tasks = [(kind, wl, cfg["width"], g) for wl in cfg["wavelengths"] for g in GAPS]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        dn = list(pool.map(_splitting, tasks))
    n = len(GAPS)
    return {
        "gaps": list(GAPS),
        "dn": {
            str(wl): dn[i * n : (i + 1) * n] for i, wl in enumerate(cfg["wavelengths"])
        },
    }


def fit(raw: dict) -> dict:
    """Least-squares fit of log dn = a0 + a1*dwl - gap * (b0 + b1*dwl)."""
    wls = np.array([float(w) for w in raw["dn"]])
    wl0 = float(wls.mean())
    rows, rhs = [], []
    for w in raw["dn"]:
        dwl = float(w) - wl0
        for gap, dn in zip(raw["gaps"], raw["dn"][w], strict=True):
            rows.append([1.0, dwl, -gap, -gap * dwl])
            rhs.append(np.log(dn))
    a, b = np.array(rows), np.array(rhs)
    coef, *_ = np.linalg.lstsq(a, b, rcond=None)
    return {
        "wl0": wl0,
        "coef": coef.tolist(),
        "max_log_residual": float(np.abs(a @ coef - b).max()),
    }


def _kappa_l(params, wl, gap):
    a0, a1, b0, b1 = params["coef"]
    dwl = wl - params["wl0"]
    return np.pi * np.exp(a0 + a1 * dwl - gap * (b0 + b1 * dwl)) / wl


def _arc_phase(params, wl, gap, radius, both_bend):
    x = np.linspace(0, min(radius, 10.0), 801)
    dy = radius - np.sqrt(np.maximum(radius**2 - x**2, 0))
    return np.trapezoid(_kappa_l(params, wl, gap + (2 if both_bend else 1) * dy), x)


def ring_kappa(params, wl, gap, radius, length_x):
    """Ring coupler: straight bus, ring bends away from it."""
    phi = _kappa_l(params, wl, gap) * length_x
    phi += 2 * _arc_phase(params, wl, gap, radius, both_bend=False)
    return np.abs(np.sin(phi))


def dc_kappa(params, wl, gap, radius, length_x, v_offset):
    """Directional coupler: both waveguides bend away (v_offset >> decay length)."""
    phi = _kappa_l(params, wl, gap) * length_x
    phi += 2 * _arc_phase(params, wl, gap, radius, both_bend=True)
    return np.abs(np.sin(phi))


def write_tables(band: str, kind: str, raw: dict) -> None:
    """Write ring and directional coupler tables for one band and cross-section."""
    params = fit(raw)
    print(
        f"{band} {kind}: decay length {1 / params['coef'][2]:.3f} um, "
        f"max log residual {params['max_log_residual']:.3f}"
    )
    suffix = "" if band == "cband" else f"_{band}"
    wls = [float(w) for w in raw["dn"]]
    attrs = {
        "description": f"{band} {kind} coupling amplitude from coupled-mode theory "
        "(samples/coupler_cmt.py). Not validated by the foundry.",
    }
    for name, func in [
        ("coupler_racetrack", ring_kappa),
        ("directional_coupler", dc_kappa),
    ]:
        coords = {"wavelength": np.linspace(min(wls), max(wls), 21), **GRIDS[name]}
        grids = np.meshgrid(*coords.values(), indexing="ij")
        kappa = np.vectorize(func, excluded={0})(params, *grids)
        xr.DataArray(kappa, coords=coords, dims=list(coords), attrs=attrs).to_netcdf(
            MODELS / f"{name}_{kind}{suffix}.nc"
        )


if __name__ == "__main__":
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    if "--from-cache" in sys.argv:
        targets = [tuple(key.split(",")) for key in cache]
    else:
        args = sys.argv[1:]
        targets = list(zip(args[::2], args[1::2], strict=True)) or list(CONFIGS)
        for band, kind in targets:
            cache[f"{band},{kind}"] = solve_splitting(band, kind)
            CACHE.write_text(json.dumps(cache, indent=1))
    for band, kind in targets:
        write_tables(band, kind, cache[f"{band},{kind}"])
