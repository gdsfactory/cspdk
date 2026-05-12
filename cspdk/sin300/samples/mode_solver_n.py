"""Nitride mode solver."""

nm = 1e-3

if __name__ == "__main__":
    import gplugins.tidy3d as gt

    wg_nc = gt.modes.Waveguide(
        wavelength=1.55,
        core_width=1.20,
        core_thickness=0.3,
        slab_thickness=0.0,
        core_material="sin",
        clad_material="sio2",
        group_index_step=10 * nm,
    )

    print("wg_nc_ng = ", wg_nc.n_group)
    print("wg_nc_neff = ", wg_nc.n_eff)

    wg_no = gt.modes.Waveguide(
        wavelength=1.31,
        core_width=0.95,
        core_thickness=0.3,
        slab_thickness=0.0,
        core_material="sin",
        clad_material="sio2",
        group_index_step=10 * nm,
    )

    print("wg_no_ng = ", wg_no.n_group)
    print("wg_no_neff = ", wg_no.n_eff)
