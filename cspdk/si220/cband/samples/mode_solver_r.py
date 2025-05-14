"""Silicon rib mode solver."""

nm = 1e-3

if __name__ == "__main__":
    import gplugins.tidy3d as gt

    wg_rib = gt.modes.Waveguide(
        wavelength=1.55,
        core_width=0.45,
        core_thickness=0.22,
        slab_thickness=0.0,
        core_material="si",
        clad_material="sio2",
        group_index_step=10 * nm,
    )

    print("wg_rib_ng = ", wg_rib.n_group)
    print("wg_rib_neff = ", wg_rib.n_eff)

    wg_ro = gt.modes.Waveguide(
        wavelength=1.31,
        core_width=0.4,
        core_thickness=0.22,
        slab_thickness=0.1,
        core_material="si",
        clad_material="sio2",
        group_index_step=10 * nm,
    )

    print("wg_ro_ng = ", wg_ro.n_group)
    print("wg_ro_neff = ", wg_ro.n_eff)
