"""Silicon rib mode solver."""

nm = 1e-3

if __name__ == "__main__":
    import gplugins.tidy3d as gt

    wg_rc = gt.modes.Waveguide(
        wavelength=1.55,
        core_width=0.45,
        core_thickness=0.5,
        slab_thickness=0.2,
        core_material="si",
        clad_material="sio2",
        group_index_step=10 * nm,
    )

    print("wg_rc_ng = ", wg_rc.n_group)
    print("wg_rc_neff = ", wg_rc.n_eff)

    wg_ro = gt.modes.Waveguide(
        wavelength=1.31,
        core_width=0.4,
        core_thickness=0.5,
        slab_thickness=0.2,
        core_material="si",
        clad_material="sio2",
        group_index_step=10 * nm,
    )

    print("wg_ro_ng = ", wg_ro.n_group)
    print("wg_ro_neff = ", wg_ro.n_eff)
