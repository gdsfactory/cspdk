"""Mode solve for the suspended-Si 3.8um TE waveguide (source of models.py).

Cross-section (Cornerstone suspended-Si library + MPW #7 guidelines):

- 450 nm Si (500 nm SOI thinned by the HF release), air above and below
  (the BOX is undercut by ~8 um, so the 3 um BOX is gone under the guide);
- 1.5 um wide core;
- 3.5 um side cladding on each side made of 0.3 um etched slots at 0.55 um
  pitch (Si fill factor 0.25 / 0.55 = 0.4545), i.e. a sub-wavelength grating
  (SWG) periodic along the propagation direction.

Approximations:

- The SWG cladding is replaced by a homogeneous zeroth-order effective medium
  for fields parallel to the slot walls (the TE-dominant Ey and Ez):
  n_swg^2 = f n_Si^2 + (1 - f). Bloch-mode / higher-order EMT corrections
  are ignored.
- The un-etched Si beyond the 3.5 um SWG cladding is not modelled (replaced
  by air); the mode has decayed by orders of magnitude across the cladding.
- Si dispersion: Chandler-Horowitz & Amirtharaj, J. Appl. Phys. 97 (2005),
  valid 1.36-11 um.

Run with `python -m cspdk.si_sus.samples.mode_solver`; it prints the values
hard-coded in `cspdk.si_sus.models`.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np

SI_THICKNESS = 0.45
CORE_WIDTH = 1.5
SWG_WIDTH = 3.5
SWG_FILL_SI = 0.25 / 0.55


def n_si(wl: float) -> float:
    """Return the refractive index of crystalline Si (wl in um)."""
    return float(np.sqrt(11.67316 + 1 / wl**2 + 0.004482633 / (wl**2 - 1.108205**2)))


def n_swg(wl: float) -> float:
    """Return the zeroth-order effective index of the slotted cladding."""
    return float(np.sqrt(SWG_FILL_SI * n_si(wl) ** 2 + (1 - SWG_FILL_SI)))


def neff(wl: float, core_width: float = CORE_WIDTH, with_swg: bool = True) -> float:
    """Return the fundamental TE effective index at wavelength wl (um).

    Args:
        wl: wavelength in um.
        core_width: core width in um.
        with_swg: model the slotted cladding as an effective medium; if False
            the core is suspended in air.
    """
    import shapely  # noqa: PLC0415
    from femwell.maxwell.waveguide import compute_modes  # noqa: PLC0415
    from femwell.mesh import mesh_from_OrderedDict  # noqa: PLC0415
    from skfem import Basis, ElementTriP0  # noqa: PLC0415
    from skfem.io.meshio import from_meshio  # noqa: PLC0415

    t, w, s = SI_THICKNESS, core_width, SWG_WIDTH
    shapes = OrderedDict(
        core=shapely.box(-w / 2, 0, w / 2, t),
        swg=shapely.box(-w / 2 - s, 0, w / 2 + s, t),
        air=shapely.box(-w / 2 - s - 3, -2.5, w / 2 + s + 3, t + 2.5),
    )
    # neff converges to ~1e-3 (order-2 elements agree within 0.0011)
    resolutions = {
        "core": {"resolution": 0.025, "distance": 0.5},
        "swg": {"resolution": 0.06, "distance": 0.5},
    }
    mesh = from_meshio(
        mesh_from_OrderedDict(shapes, resolutions, default_resolution_max=0.5)
    )
    basis0 = Basis(mesh, ElementTriP0())
    epsilon = basis0.zeros() + 1.0
    epsilon[basis0.get_dofs(elements="swg")] = (n_swg(wl) if with_swg else 1.0) ** 2
    epsilon[basis0.get_dofs(elements="core")] = n_si(wl) ** 2
    modes = compute_modes(basis0, epsilon, wavelength=wl, num_modes=2, n_guess=3.0)
    te = [m for m in modes if m.te_fraction > 0.5]
    return float(max(te, key=lambda m: m.n_eff.real).n_eff.real)


def ng(wl: float, dwl: float = 0.01, **kwargs) -> float:
    """Return the group index from a central difference of neff.

    Args:
        wl: wavelength in um.
        dwl: wavelength step in um.
        kwargs: passed to `neff`.
    """
    n1, n0, n2 = (neff(x, **kwargs) for x in (wl - dwl, wl, wl + dwl))
    return n0 - wl * (n2 - n1) / (2 * dwl)


if __name__ == "__main__":
    wl0 = 3.8
    print(f"n_Si({wl0}) = {n_si(wl0):.4f}, n_swg = {n_swg(wl0):.4f}")
    for with_swg in (True, False):
        label = "SWG cladding" if with_swg else "core in air"
        print(
            f"{label}: neff = {neff(wl0, with_swg=with_swg):.4f}, "
            f"ng = {ng(wl0, with_swg=with_swg):.4f}"
        )
