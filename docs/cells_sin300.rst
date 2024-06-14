

Cells SiN300
=============================


array
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.array

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.array(component='pad', spacing=(150.0, 150.0), columns=6, rows=1, add_ports=True, centered=False)
  c.plot()



bend_euler
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.bend_euler

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.bend_euler(angle=90.0, p=0.5, with_arc_floorplan=True, cross_section='xs_nc', allow_min_radius_violation=False)
  c.plot()



bend_s
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.bend_s

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.bend_s(size=(11.0, 1.8), npoints=99, cross_section='xs_nc', allow_min_radius_violation=False)
  c.plot()



bend_sc
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.bend_sc

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.bend_sc(angle=90.0, p=0.5, with_arc_floorplan=True, cross_section='xs_nc', allow_min_radius_violation=False)
  c.plot()



bend_so
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.bend_so

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.bend_so(angle=90.0, p=0.5, with_arc_floorplan=True, cross_section='xs_no', allow_min_radius_violation=False)
  c.plot()



coupler
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.coupler

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.coupler(gap=0.234, length=20.0, dy=4.0, dx=15.0, cross_section='xs_nc')
  c.plot()



coupler_nc
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.coupler_nc

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.coupler_nc(gap=0.234, length=20.0, dy=4.0, dx=15.0, cross_section='xs_nc')
  c.plot()



coupler_no
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.coupler_no

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.coupler_no(gap=0.234, length=20.0, dy=4.0, dx=15.0, cross_section='xs_no')
  c.plot()



die_rc
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.die_rc

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.die_rc(size=(11470.0, 4900.0), ngratings=14, npads=31, grating_pitch=250.0, pad_pitch=300.0, grating_coupler='grating_coupler_rectangular_rc', cross_section='xs_rc', pad='pad')
  c.plot()



die_ro
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.die_ro

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.die_ro(size=(11470.0, 4900.0), ngratings=14, npads=31, grating_pitch=250.0, pad_pitch=300.0, grating_coupler='grating_coupler_rectangular_ro', cross_section='xs_ro', pad='pad')
  c.plot()



die_sc
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.die_sc

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.die_sc(size=(11470.0, 4900.0), ngratings=14, npads=31, grating_pitch=250.0, pad_pitch=300.0, grating_coupler='grating_coupler_rectangular_sc', cross_section='xs_sc', pad='pad')
  c.plot()



die_so
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.die_so

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.die_so(size=(11470.0, 4900.0), ngratings=14, npads=31, grating_pitch=250.0, pad_pitch=300.0, grating_coupler='grating_coupler_rectangular_so', cross_section='xs_so', pad='pad')
  c.plot()



grating_coupler_array
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.grating_coupler_array

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.grating_coupler_array(grating_coupler='grating_coupler_rectangular_nc', pitch=127, n=6, port_name='o1', rotation=-90, with_loopback=False, cross_section='xs_sc', straight_to_grating_spacing=10.0, centered=True)
  c.plot()



grating_coupler_elliptical_trenches
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.grating_coupler_elliptical_trenches

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.grating_coupler_elliptical_trenches(polarization='te', taper_length=16.6, taper_angle=30.0, trenches_extra_angle=9.0, wavelength=1.55, fiber_angle=20.0, grating_line_width=0.343, neff=1.6, ncladding=1.443, layer_trench=<LayerMapCornerstone.GRA: 4>, p_start=26, n_periods=30, end_straight_length=0.2, cross_section='xs_nc')
  c.plot()



grating_coupler_elliptical_trenches_nc
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.grating_coupler_elliptical_trenches_nc

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.grating_coupler_elliptical_trenches_nc(polarization='te', taper_length=16.6, taper_angle=30.0, trenches_extra_angle=9.0, wavelength=1.55, fiber_angle=20.0, grating_line_width=0.343, neff=1.6, ncladding=1.443, layer_trench=<LayerMapCornerstone.GRA: 4>, p_start=26, n_periods=30, end_straight_length=0.2, cross_section='xs_nc')
  c.plot()



grating_coupler_elliptical_trenches_no
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.grating_coupler_elliptical_trenches_no

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.grating_coupler_elliptical_trenches_no(polarization='te', taper_length=16.6, taper_angle=30.0, trenches_extra_angle=9.0, wavelength=1.55, fiber_angle=20.0, grating_line_width=0.343, neff=1.6, ncladding=1.443, layer_trench=<LayerMapCornerstone.GRA: 4>, p_start=26, n_periods=30, end_straight_length=0.2, cross_section='xs_no')
  c.plot()



grating_coupler_rectangular
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.grating_coupler_rectangular

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.grating_coupler_rectangular(n_periods=30, period=0.75, fill_factor=0.5, width_grating=11.0, length_taper=200.0, polarization='te', wavelength=1.55, layer_slab=<LayerMapCornerstone.NITRIDE: 7>, layer_grating=<LayerMapCornerstone.NITRIDE_ETCH: 8>, fiber_angle=20.0, slab_xmin=-1.0, slab_offset=0.0, cross_section='xs_nc')
  c.plot()



grating_coupler_rectangular_nc
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.grating_coupler_rectangular_nc

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.grating_coupler_rectangular_nc(n_periods=30, period=0.66, fill_factor=0.5, width_grating=11.0, length_taper=200.0, polarization='te', wavelength=1.55, layer_slab=<LayerMapCornerstone.NITRIDE: 7>, layer_grating=<LayerMapCornerstone.NITRIDE_ETCH: 8>, fiber_angle=20.0, slab_xmin=-1.0, slab_offset=0.0, cross_section='xs_nc')
  c.plot()



grating_coupler_rectangular_no
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.grating_coupler_rectangular_no

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.grating_coupler_rectangular_no(n_periods=30, period=0.964, fill_factor=0.5, width_grating=11.0, length_taper=200.0, polarization='te', wavelength=1.55, layer_slab=<LayerMapCornerstone.NITRIDE: 7>, layer_grating=<LayerMapCornerstone.NITRIDE_ETCH: 8>, fiber_angle=20.0, slab_xmin=-1.0, slab_offset=0.0, cross_section='xs_no')
  c.plot()



mmi1x2
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.mmi1x2

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.mmi1x2(width_taper=5.5, length_taper=50.0, length_mmi=5.5, width_mmi=12.0, gap_mmi=0.25, cross_section='xs_nc')
  c.plot()



mmi1x2_nc
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.mmi1x2_nc

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.mmi1x2_nc(width_taper=5.5, length_taper=50.0, length_mmi=64.7, width_mmi=12.0, gap_mmi=0.4, cross_section='xs_nc')
  c.plot()



mmi1x2_no
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.mmi1x2_no

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.mmi1x2_no(width_taper=5.5, length_taper=50.0, length_mmi=42.0, width_mmi=12.0, gap_mmi=0.4, cross_section='xs_no')
  c.plot()



mmi2x2
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.mmi2x2

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.mmi2x2(width_taper=5.5, length_taper=50.0, length_mmi=5.5, width_mmi=12.0, gap_mmi=0.25, cross_section='xs_nc')
  c.plot()



mmi2x2_nc
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.mmi2x2_nc

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.mmi2x2_nc(width_taper=5.5, length_taper=50.0, length_mmi=232.0, width_mmi=12.0, gap_mmi=0.4, cross_section='xs_nc')
  c.plot()



mmi2x2_no
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.mmi2x2_no

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.mmi2x2_no(width_taper=5.5, length_taper=50.0, length_mmi=126.0, width_mmi=12.0, gap_mmi=0.4, cross_section='xs_no')
  c.plot()



mzi
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.mzi

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.mzi(delta_length=10.0, length_y=2.0, length_x=0.1, splitter='mmi1x2', with_splitter=True, port_e1_splitter='o2', port_e0_splitter='o3', port_e1_combiner='o2', port_e0_combiner='o3', nbends=2, cross_section='xs_nc', mirror_bot=False, add_optical_ports_arms=False, min_length=0.01, auto_rename_ports=True)
  c.plot()



mzi_nc
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.mzi_nc

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.mzi_nc(delta_length=10.0, length_y=2.0, length_x=0.1, splitter='mmi1x2', with_splitter=True, port_e1_splitter='o2', port_e0_splitter='o3', port_e1_combiner='o2', port_e0_combiner='o3', nbends=2, cross_section='xs_nc', mirror_bot=False, add_optical_ports_arms=False, min_length=0.01, auto_rename_ports=True)
  c.plot()



mzi_no
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.mzi_no

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.mzi_no(delta_length=10.0, length_y=2.0, length_x=0.1, splitter='mmi1x2', with_splitter=True, port_e1_splitter='o2', port_e0_splitter='o3', port_e1_combiner='o2', port_e0_combiner='o3', nbends=2, cross_section='xs_no', mirror_bot=False, add_optical_ports_arms=False, min_length=0.01, auto_rename_ports=True)
  c.plot()



pad
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.pad

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.pad(size=(100.0, 100.0), layer='PAD', port_inclusion=0, port_orientation=0)
  c.plot()



rectangle
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.rectangle

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.rectangle(size=(4.0, 2.0), layer=<LayerMapCornerstone.FLOORPLAN: 2>, centered=False, port_type='electrical', port_orientations=(180, 90, 0, -90))
  c.plot()



straight
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.straight

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.straight(length=10.0, npoints=2, cross_section='xs_nc')
  c.plot()



straight_nc
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.straight_nc

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.straight_nc(length=10.0, npoints=2, cross_section='xs_nc')
  c.plot()



straight_no
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.straight_no

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.straight_no(length=10.0, npoints=2, cross_section='xs_no')
  c.plot()



taper
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.taper

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.taper(length=10.0, width1=0.5, with_two_ports=True, cross_section='xs_nc', port_names=('o1', 'o2'), port_types=('optical', 'optical'), with_bbox=True)
  c.plot()



taper_nc
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.taper_nc

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.taper_nc(length=10.0, width1=0.5, with_two_ports=True, cross_section='xs_nc', port_names=('o1', 'o2'), port_types=('optical', 'optical'), with_bbox=True)
  c.plot()



taper_no
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.taper_no

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.taper_no(length=10.0, width1=0.5, with_two_ports=True, cross_section='xs_no', port_names=('o1', 'o2'), port_types=('optical', 'optical'), with_bbox=True)
  c.plot()



wire_corner
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.wire_corner

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.wire_corner(cross_section='metal_routing')
  c.plot()
