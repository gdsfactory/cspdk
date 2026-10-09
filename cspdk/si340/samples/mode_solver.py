"""Mode solver for the waveguide indices used in ``cspdk.si340.models``.

Stack from the CORNERSTONE 340 nm SOI (49th call) design guidelines: 340 nm
Si on a 2 um BOX with a 1 um SiO2 top cladding; strips are fully etched and
ribs are etched 140 nm (200 nm slab).
"""

nm = 1e-3

if __name__ == "__main__":
    import gplugins.tidy3d as gt

    for name, wavelength, width, slab in (
        ("sc", 1.55, 0.45, 0.0),
        ("so", 1.31, 0.40, 0.0),
        ("rc", 1.55, 0.80, 200 * nm),
    ):
        wg = gt.modes.Waveguide(
            wavelength=wavelength,
            core_width=width,
            core_thickness=340 * nm,
            slab_thickness=slab,
            core_material="Si",
            clad_material="SiO2",
            box_material="SiO2",
            box_thickness=2.0,
            clad_thickness=1.0,
            num_modes=4,
            group_index_step=10 * nm,
            grid_resolution=40,
        )
        print(f"wg_{name}_neff = ", wg.n_eff[0])
        print(f"wg_{name}_ng = ", wg.n_group[0])
