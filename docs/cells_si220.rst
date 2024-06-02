

Cells Si SOI 220nm
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

  c = cspdk.si220.cells.bend_euler(angle=90.0, p=0.5, with_arc_floorplan=True, cross_section='xs_sc', allow_min_radius_violation=False)
  c.plot()



bend_rc
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.bend_rc

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.bend_rc(angle=90.0, p=0.5, with_arc_floorplan=True, cross_section='xs_rc', allow_min_radius_violation=False)
  c.plot()



bend_ro
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.bend_ro

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.bend_ro(angle=90.0, p=0.5, with_arc_floorplan=True, cross_section='xs_ro', allow_min_radius_violation=False)
  c.plot()



bend_s
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.bend_s

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.bend_s(size=(11.0, 1.8), npoints=99, cross_section='xs_sc', allow_min_radius_violation=False)
  c.plot()



bend_sc
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.bend_sc

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.bend_sc(angle=90.0, p=0.5, with_arc_floorplan=True, cross_section='xs_sc', allow_min_radius_violation=False)
  c.plot()



bend_so
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.bend_so

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.bend_so(angle=90.0, p=0.5, with_arc_floorplan=True, cross_section='xs_so', allow_min_radius_violation=False)
  c.plot()



couple_symmetric
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.couple_symmetric

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.couple_symmetric(gap=0.234, dy=4.0, dx=10.0, cross_section='xs_sc')
  c.plot()



coupler
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.coupler

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.coupler(gap=0.234, length=20.0, dy=4.0, dx=10.0, cross_section='xs_sc')
  c.plot()



coupler_rc
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.coupler_rc

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.coupler_rc(gap=0.234, length=20.0, dy=4.0, dx=15, cross_section='xs_rc')
  c.plot()



coupler_ro
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.coupler_ro

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.coupler_ro(gap=0.234, length=20.0, dy=4.0, dx=15, cross_section='xs_ro')
  c.plot()



coupler_sc
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.coupler_sc

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.coupler_sc(gap=0.234, length=20.0, dy=4.0, dx=10.0, cross_section='xs_sc')
  c.plot()



coupler_so
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.coupler_so

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.coupler_so(gap=0.234, length=20.0, dy=4.0, dx=10.0, cross_section='xs_so')
  c.plot()



crossing_rc
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.crossing_rc

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.crossing_rc()
  c.plot()



crossing_sc
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.crossing_sc

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.crossing_sc()
  c.plot()



crossing_so
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.crossing_so

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.crossing_so()
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

  c = cspdk.si220.cells.grating_coupler_array(grating_coupler='grating_coupler_rectangular_sc', pitch=127, n=6, port_name='o1', rotation=-90, with_loopback=False, cross_section='xs_sc', straight_to_grating_spacing=10.0, centered=True)
  c.plot()



grating_coupler_elliptical_trenches
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.grating_coupler_elliptical_trenches

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.grating_coupler_elliptical_trenches(polarization='te', taper_length=16.6, taper_angle=30.0, trenches_extra_angle=9.0, wavelength=1.53, fiber_angle=15.0, grating_line_width=0.315, neff=2.638, ncladding=1.443, layer_trench=<LayerMapCornerstone.GRA: 4>, p_start=26, n_periods=30, end_straight_length=0.2, cross_section='xs_sc')
  c.plot()



grating_coupler_elliptical_trenches_sc
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.grating_coupler_elliptical_trenches_sc

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.grating_coupler_elliptical_trenches_sc(polarization='te', taper_length=16.6, taper_angle=30.0, trenches_extra_angle=9.0, wavelength=1.55, fiber_angle=15.0, grating_line_width=0.315, neff=2.638, ncladding=1.443, layer_trench=<LayerMapCornerstone.GRA: 4>, p_start=26, n_periods=30, end_straight_length=0.2, cross_section='xs_sc')
  c.plot()



grating_coupler_elliptical_trenches_so
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.grating_coupler_elliptical_trenches_so

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.grating_coupler_elliptical_trenches_so(polarization='te', taper_length=16.6, taper_angle=30.0, trenches_extra_angle=9.0, wavelength=1.31, fiber_angle=15.0, grating_line_width=0.25, neff=2.638, ncladding=1.443, layer_trench=<LayerMapCornerstone.GRA: 4>, p_start=26, n_periods=30, end_straight_length=0.2, cross_section='xs_so')
  c.plot()



grating_coupler_rectangular
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.grating_coupler_rectangular

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.grating_coupler_rectangular(n_periods=30, period=0.63, fill_factor=0.5, width_grating=11.0, length_taper=350.0, polarization='te', wavelength=1.55, layer_slab=<LayerMapCornerstone.WG: 0>, layer_grating=<LayerMapCornerstone.GRA: 4>, fiber_angle=10.0, slab_xmin=-1.0, slab_offset=0.0, cross_section='xs_sc')
  c.plot()



grating_coupler_rectangular_rc
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.grating_coupler_rectangular_rc

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.grating_coupler_rectangular_rc(n_periods=60, period=0.5, fill_factor=0.5, width_grating=11.0, length_taper=350.0, polarization='te', wavelength=1.55, layer_slab=<LayerMapCornerstone.WG: 0>, layer_grating=<LayerMapCornerstone.GRA: 4>, fiber_angle=10.0, slab_xmin=-1.0, slab_offset=0.0, cross_section='xs_rc')
  c.plot()



grating_coupler_rectangular_ro
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.grating_coupler_rectangular_ro

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.grating_coupler_rectangular_ro(n_periods=80, period=0.5, fill_factor=0.5, width_grating=11.0, length_taper=350.0, polarization='te', wavelength=1.55, layer_slab=<LayerMapCornerstone.WG: 0>, layer_grating=<LayerMapCornerstone.GRA: 4>, fiber_angle=10.0, slab_xmin=-1.0, slab_offset=0.0, cross_section='xs_ro')
  c.plot()



grating_coupler_rectangular_sc
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.grating_coupler_rectangular_sc

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.grating_coupler_rectangular_sc(n_periods=60, period=0.63, fill_factor=0.5, width_grating=11.0, length_taper=350.0, polarization='te', wavelength=1.55, layer_slab=<LayerMapCornerstone.WG: 0>, layer_grating=<LayerMapCornerstone.GRA: 4>, fiber_angle=10.0, slab_xmin=-1.0, slab_offset=0.0, cross_section='xs_sc')
  c.plot()



grating_coupler_rectangular_so
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.grating_coupler_rectangular_so

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.grating_coupler_rectangular_so(n_periods=80, period=0.5, fill_factor=0.5, width_grating=11.0, length_taper=350.0, polarization='te', wavelength=1.31, layer_slab=<LayerMapCornerstone.WG: 0>, layer_grating=<LayerMapCornerstone.GRA: 4>, fiber_angle=10.0, slab_xmin=-1.0, slab_offset=0.0, cross_section='xs_so')
  c.plot()



heater
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.heater

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.heater()
  c.plot()



mmi1x2
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.mmi1x2

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.mmi1x2(width_taper=1.5, length_taper=20.0, length_mmi=5.5, width_mmi=6.0, gap_mmi=0.25, cross_section='xs_sc')
  c.plot()



mmi1x2_rc
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.mmi1x2_rc

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.mmi1x2_rc(width_taper=1.5, length_taper=20.0, length_mmi=32.7, width_mmi=6.0, gap_mmi=1.64, cross_section='xs_rc')
  c.plot()



mmi1x2_ro
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.mmi1x2_ro

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.mmi1x2_ro(width_taper=1.5, length_taper=20.0, length_mmi=40.8, width_mmi=6.0, gap_mmi=1.55, cross_section='xs_ro')
  c.plot()



mmi1x2_sc
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.mmi1x2_sc

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.mmi1x2_sc(width_taper=1.5, length_taper=20.0, length_mmi=31.8, width_mmi=6.0, gap_mmi=1.64, cross_section='xs_sc')
  c.plot()



mmi1x2_so
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.mmi1x2_so

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.mmi1x2_so(width_taper=1.5, length_taper=20.0, length_mmi=40.1, width_mmi=6.0, gap_mmi=1.55, cross_section='xs_so')
  c.plot()



mmi2x2
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.mmi2x2

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.mmi2x2(width_taper=1.5, length_taper=20.0, length_mmi=5.5, width_mmi=6.0, gap_mmi=0.25, cross_section='xs_sc')
  c.plot()



mmi2x2_rc
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.mmi2x2_rc

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.mmi2x2_rc(width_taper=1.5, length_taper=20.0, length_mmi=44.8, width_mmi=6.0, gap_mmi=0.53, cross_section='xs_rc')
  c.plot()



mmi2x2_ro
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.mmi2x2_ro

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.mmi2x2_ro(width_taper=1.5, length_taper=20.0, length_mmi=55.0, width_mmi=6.0, gap_mmi=0.53, cross_section='xs_ro')
  c.plot()



mmi2x2_sc
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.mmi2x2_sc

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.mmi2x2_sc(width_taper=1.5, length_taper=20.0, length_mmi=42.5, width_mmi=6.0, gap_mmi=0.5, cross_section='xs_sc')
  c.plot()



mmi2x2_so
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.mmi2x2_so

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.mmi2x2_so(width_taper=1.5, length_taper=20.0, length_mmi=53.5, width_mmi=6.0, gap_mmi=0.53, cross_section='xs_so')
  c.plot()



mzi_rc
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.mzi_rc

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.mzi_rc(delta_length=10.0, length_y=2.0, length_x=0.1, with_splitter=True, port_e1_splitter='o2', port_e0_splitter='o3', port_e1_combiner='o2', port_e0_combiner='o3', nbends=2, cross_section='xs_rc', mirror_bot=False, add_optical_ports_arms=False, min_length=0.01, auto_rename_ports=True)
  c.plot()



mzi_ro
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.mzi_ro

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.mzi_ro(delta_length=10.0, length_y=2.0, length_x=0.1, with_splitter=True, port_e1_splitter='o2', port_e0_splitter='o3', port_e1_combiner='o2', port_e0_combiner='o3', nbends=2, cross_section='xs_ro', mirror_bot=False, add_optical_ports_arms=False, min_length=0.01, auto_rename_ports=True)
  c.plot()



mzi_sc
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.mzi_sc

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.mzi_sc(delta_length=10.0, length_y=2.0, length_x=0.1, with_splitter=True, port_e1_splitter='o2', port_e0_splitter='o3', port_e1_combiner='o3', port_e0_combiner='o4', nbends=2, cross_section='xs_sc', mirror_bot=False, add_optical_ports_arms=False, min_length=0.01, auto_rename_ports=True)
  c.plot()



mzi_so
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.mzi_so

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.mzi_so(delta_length=10.0, length_y=2.0, length_x=0.1, with_splitter=True, port_e1_splitter='o2', port_e0_splitter='o3', port_e1_combiner='o2', port_e0_combiner='o4', nbends=2, cross_section='xs_so', mirror_bot=False, add_optical_ports_arms=False, min_length=0.01, auto_rename_ports=True)
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

  c = cspdk.si220.cells.straight(length=10.0, npoints=2, cross_section='xs_sc')
  c.plot()



straight_rc
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.straight_rc

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.straight_rc(length=10.0, npoints=2, cross_section='xs_rc')
  c.plot()



straight_ro
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.straight_ro

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.straight_ro(length=10.0, npoints=2, cross_section='xs_ro')
  c.plot()



straight_sc
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.straight_sc

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.straight_sc(length=10.0, npoints=2, cross_section='xs_sc')
  c.plot()



straight_so
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.straight_so

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.straight_so(length=10.0, npoints=2, cross_section='xs_so')
  c.plot()



taper
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.taper

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.taper(length=10.0, width1=0.5, with_two_ports=True, cross_section='xs_sc', port_names=('o1', 'o2'), port_types=('optical', 'optical'), with_bbox=True)
  c.plot()



taper_rc
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.taper_rc

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.taper_rc(length=10.0, width1=0.5, with_two_ports=True, cross_section='xs_rc', port_names=('o1', 'o2'), port_types=('optical', 'optical'), with_bbox=True)
  c.plot()



taper_ro
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.taper_ro

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.taper_ro(length=10.0, width1=0.5, with_two_ports=True, cross_section='xs_ro', port_names=('o1', 'o2'), port_types=('optical', 'optical'), with_bbox=True)
  c.plot()



taper_sc
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.taper_sc

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.taper_sc(length=10.0, width1=0.5, with_two_ports=True, cross_section='xs_sc', port_names=('o1', 'o2'), port_types=('optical', 'optical'), with_bbox=True)
  c.plot()



taper_so
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.taper_so

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.taper_so(length=10.0, width1=0.5, with_two_ports=True, cross_section='xs_so', port_names=('o1', 'o2'), port_types=('optical', 'optical'), with_bbox=True)
  c.plot()



taper_strip_to_ridge
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.taper_strip_to_ridge

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.taper_strip_to_ridge(length=10, width1=0.5, width2=0.5, w_slab1=0.2, w_slab2=10.45, layer_wg=<LayerMapCornerstone.WG: 0>, layer_slab=<LayerMapCornerstone.SLAB: 1>, cross_section='xs_sc', use_slab_port=False)
  c.plot()



trans_sc_rc10
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.trans_sc_rc10

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.trans_sc_rc10(length=10, width1=0.5, width2=0.5, w_slab1=0.2, w_slab2=10.45, layer_wg=<LayerMapCornerstone.WG: 0>, layer_slab=<LayerMapCornerstone.SLAB: 1>, cross_section='xs_sc', use_slab_port=False)
  c.plot()



trans_sc_rc20
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.trans_sc_rc20

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.trans_sc_rc20(length=20, width1=0.5, width2=0.5, w_slab1=0.2, w_slab2=10.45, layer_wg=<LayerMapCornerstone.WG: 0>, layer_slab=<LayerMapCornerstone.SLAB: 1>, cross_section='xs_sc', use_slab_port=False)
  c.plot()



trans_sc_rc50
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.trans_sc_rc50

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.trans_sc_rc50(length=50, width1=0.5, width2=0.5, w_slab1=0.2, w_slab2=10.45, layer_wg=<LayerMapCornerstone.WG: 0>, layer_slab=<LayerMapCornerstone.SLAB: 1>, cross_section='xs_sc', use_slab_port=False)
  c.plot()



wire_corner
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.wire_corner

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.wire_corner(cross_section='metal_routing')
  c.plot()
