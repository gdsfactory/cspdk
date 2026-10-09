"""Silicon rib mode solver for the indices used in ``cspdk.si500.models``.

Stack from the CORNERSTONE 500 nm SOI (42nd call) design guidelines: 500 nm
Si, 300 nm rib etch (200 nm slab), 3 um BOX and 2 um SiO2 top cladding.
"""

nm = 1e-3

if __name__ == "__main__":
    import gplugins.tidy3d as gt

    for name, wavelength, width in (("rc", 1.55, 0.45), ("ro", 1.31, 0.40)):
        wg = gt.modes.Waveguide(
            wavelength=wavelength,
            core_width=width,
            core_thickness=500 * nm,
            slab_thickness=200 * nm,
            core_material="Si",
            clad_material="SiO2",
            box_material="SiO2",
            box_thickness=3.0,
            clad_thickness=2.0,
            num_modes=4,
            group_index_step=10 * nm,
            grid_resolution=40,
        )
        print(f"wg_{name}_neff = ", wg.n_eff[0])
        print(f"wg_{name}_ng = ", wg.n_group[0])
