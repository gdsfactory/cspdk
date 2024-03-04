

Here are the components available in the PDK


Cells
=============================


bend_nc
----------------------------------------------------

.. autofunction:: cspdk.cells.bend_nc

.. plot::
  :include-source:

  import cspdk

  c = cspdk.cells.bend_nc(angle=90.0, p=0.5, with_arc_floorplan=True, direction='ccw', add_pins=True)
  c.plot()



bend_no
----------------------------------------------------

.. autofunction:: cspdk.cells.bend_no

.. plot::
  :include-source:

  import cspdk

  c = cspdk.cells.bend_no(angle=90.0, p=0.5, with_arc_floorplan=True, direction='ccw', add_pins=True)
  c.plot()



bend_rc
----------------------------------------------------

.. autofunction:: cspdk.cells.bend_rc

.. plot::
  :include-source:

  import cspdk

  c = cspdk.cells.bend_rc(angle=90.0, p=0.5, with_arc_floorplan=True, direction='ccw', add_pins=True)
  c.plot()



bend_ro
----------------------------------------------------

.. autofunction:: cspdk.cells.bend_ro

.. plot::
  :include-source:

  import cspdk

  c = cspdk.cells.bend_ro(angle=90.0, p=0.5, with_arc_floorplan=True, direction='ccw', add_pins=True)
  c.plot()



bend_sc
----------------------------------------------------

.. autofunction:: cspdk.cells.bend_sc

.. plot::
  :include-source:

  import cspdk

  c = cspdk.cells.bend_sc(angle=90.0, p=0.5, with_arc_floorplan=True, direction='ccw', add_pins=True)
  c.plot()



bend_so
----------------------------------------------------

.. autofunction:: cspdk.cells.bend_so

.. plot::
  :include-source:

  import cspdk

  c = cspdk.cells.bend_so(angle=90.0, p=0.5, with_arc_floorplan=True, direction='ccw', add_pins=True)
  c.plot()



crossing_rc
----------------------------------------------------

.. autofunction:: cspdk.cells.crossing_rc

.. plot::
  :include-source:

  import cspdk

  c = cspdk.cells.crossing_rc()
  c.plot()



crossing_sc
----------------------------------------------------

.. autofunction:: cspdk.cells.crossing_sc

.. plot::
  :include-source:

  import cspdk

  c = cspdk.cells.crossing_sc()
  c.plot()



crossing_so
----------------------------------------------------

.. autofunction:: cspdk.cells.crossing_so

.. plot::
  :include-source:

  import cspdk

  c = cspdk.cells.crossing_so()
  c.plot()



die_nc
----------------------------------------------------

.. autofunction:: cspdk.cells.die_nc

.. plot::
  :include-source:

  import cspdk

  c = cspdk.cells.die_nc(size=(11470, 4900), ngratings=14, npads=31, grating_pitch=250, pad_pitch=300)
  c.plot()



die_no
----------------------------------------------------

.. autofunction:: cspdk.cells.die_no

.. plot::
  :include-source:

  import cspdk

  c = cspdk.cells.die_no(size=(11470, 4900), ngratings=14, npads=31, grating_pitch=250, pad_pitch=300)
  c.plot()



die_rc
----------------------------------------------------

.. autofunction:: cspdk.cells.die_rc

.. plot::
  :include-source:

  import cspdk

  c = cspdk.cells.die_rc(size=(11470, 4900), ngratings=14, npads=31, grating_pitch=250, pad_pitch=300)
  c.plot()



die_ro
----------------------------------------------------

.. autofunction:: cspdk.cells.die_ro

.. plot::
  :include-source:

  import cspdk

  c = cspdk.cells.die_ro(size=(11470, 4900), ngratings=14, npads=31, grating_pitch=250, pad_pitch=300)
  c.plot()



die_sc
----------------------------------------------------

.. autofunction:: cspdk.cells.die_sc

.. plot::
  :include-source:

  import cspdk

  c = cspdk.cells.die_sc(size=(11470, 4900), ngratings=14, npads=31, grating_pitch=250, pad_pitch=300)
  c.plot()



die_so
----------------------------------------------------

.. autofunction:: cspdk.cells.die_so

.. plot::
  :include-source:

  import cspdk

  c = cspdk.cells.die_so(size=(11470, 4900), ngratings=14, npads=31, grating_pitch=250, pad_pitch=300)
  c.plot()



gc_rectangular_nc
----------------------------------------------------

.. autofunction:: cspdk.cells.gc_rectangular_nc

.. plot::
  :include-source:

  import cspdk

  c = cspdk.cells.gc_rectangular_nc(n_periods=30, period=0.66, fill_factor=0.5, width_grating=11.0, length_taper=200, polarization='te', wavelength=1.55, layer_slab=(203, 0), layer_grating=(204, 0), fiber_angle=20, slab_xmin=-1.0, slab_offset=0)
  c.plot()



gc_rectangular_no
----------------------------------------------------

.. autofunction:: cspdk.cells.gc_rectangular_no

.. plot::
  :include-source:

  import cspdk

  c = cspdk.cells.gc_rectangular_no(n_periods=30, period=0.964, fill_factor=0.5, width_grating=11.0, length_taper=200, polarization='te', wavelength=1.55, layer_slab=(203, 0), layer_grating=(204, 0), fiber_angle=20, slab_xmin=-1.0, slab_offset=0)
  c.plot()



gc_rectangular_rc
----------------------------------------------------

.. autofunction:: cspdk.cells.gc_rectangular_rc

.. plot::
  :include-source:

  import cspdk

  c = cspdk.cells.gc_rectangular_rc(n_periods=60, period=0.5, fill_factor=0.5, width_grating=11.0, length_taper=350, polarization='te', wavelength=1.55, layer_slab=(3, 0), layer_grating=(6, 0), fiber_angle=10, slab_xmin=-1.0, slab_offset=0)
  c.plot()



gc_rectangular_ro
----------------------------------------------------

.. autofunction:: cspdk.cells.gc_rectangular_ro

.. plot::
  :include-source:

  import cspdk

  c = cspdk.cells.gc_rectangular_ro(n_periods=80, period=0.5, fill_factor=0.5, width_grating=11.0, length_taper=350, polarization='te', wavelength=1.55, layer_slab=(3, 0), layer_grating=(6, 0), fiber_angle=10, slab_xmin=-1.0, slab_offset=0)
  c.plot()



gc_rectangular_sc
----------------------------------------------------

.. autofunction:: cspdk.cells.gc_rectangular_sc

.. plot::
  :include-source:

  import cspdk

  c = cspdk.cells.gc_rectangular_sc(n_periods=60, period=0.63, fill_factor=0.5, width_grating=11.0, length_taper=350, polarization='te', wavelength=1.55, layer_slab=(3, 0), layer_grating=(6, 0), fiber_angle=10, slab_xmin=-1.0, slab_offset=0)
  c.plot()



gc_rectangular_so
----------------------------------------------------

.. autofunction:: cspdk.cells.gc_rectangular_so

.. plot::
  :include-source:

  import cspdk

  c = cspdk.cells.gc_rectangular_so(n_periods=80, period=0.5, fill_factor=0.5, width_grating=11.0, length_taper=350, polarization='te', wavelength=1.55, layer_slab=(3, 0), layer_grating=(6, 0), fiber_angle=10, slab_xmin=-1.0, slab_offset=0)
  c.plot()



heater
----------------------------------------------------

.. autofunction:: cspdk.cells.heater

.. plot::
  :include-source:

  import cspdk

  c = cspdk.cells.heater()
  c.plot()



import_gds
----------------------------------------------------

.. autofunction:: cspdk.cells.import_gds

.. plot::
  :include-source:

  import cspdk

  c = cspdk.cells.import_gds(read_metadata=False, read_metadata_json=False, keep_name_short=False, unique_names=True, max_name_length=250)
  c.plot()



mmi1x2_nc
----------------------------------------------------

.. autofunction:: cspdk.cells.mmi1x2_nc

.. plot::
  :include-source:

  import cspdk

  c = cspdk.cells.mmi1x2_nc(width_taper=5.5, length_taper=50, length_mmi=64.7, width_mmi=12, gap_mmi=0.25, with_bbox=True)
  c.plot()



mmi1x2_no
----------------------------------------------------

.. autofunction:: cspdk.cells.mmi1x2_no

.. plot::
  :include-source:

  import cspdk

  c = cspdk.cells.mmi1x2_no(width_taper=5.5, length_taper=50, length_mmi=42, width_mmi=12, gap_mmi=0.25, with_bbox=True)
  c.plot()



mmi1x2_rc
----------------------------------------------------

.. autofunction:: cspdk.cells.mmi1x2_rc

.. plot::
  :include-source:

  import cspdk

  c = cspdk.cells.mmi1x2_rc(width_taper=1.5, length_taper=20, length_mmi=32.7, width_mmi=6, gap_mmi=1.64, with_bbox=True)
  c.plot()



mmi1x2_ro
----------------------------------------------------

.. autofunction:: cspdk.cells.mmi1x2_ro

.. plot::
  :include-source:

  import cspdk

  c = cspdk.cells.mmi1x2_ro(width_taper=1.5, length_taper=20, length_mmi=40.8, width_mmi=6, gap_mmi=1.55, with_bbox=True)
  c.plot()



mmi1x2_sc
----------------------------------------------------

.. autofunction:: cspdk.cells.mmi1x2_sc

.. plot::
  :include-source:

  import cspdk

  c = cspdk.cells.mmi1x2_sc(width_taper=1.5, length_taper=20, length_mmi=31.8, width_mmi=6, gap_mmi=1.64, with_bbox=True)
  c.plot()



mmi1x2_so
----------------------------------------------------

.. autofunction:: cspdk.cells.mmi1x2_so

.. plot::
  :include-source:

  import cspdk

  c = cspdk.cells.mmi1x2_so(width_taper=1.5, length_taper=20, length_mmi=40.1, width_mmi=6, gap_mmi=1.55, with_bbox=True)
  c.plot()



mmi2x2_nc
----------------------------------------------------

.. autofunction:: cspdk.cells.mmi2x2_nc

.. plot::
  :include-source:

  import cspdk

  c = cspdk.cells.mmi2x2_nc(width_taper=5.5, length_taper=50, length_mmi=232, width_mmi=12, gap_mmi=0.25, with_bbox=True)
  c.plot()



mmi2x2_no
----------------------------------------------------

.. autofunction:: cspdk.cells.mmi2x2_no

.. plot::
  :include-source:

  import cspdk

  c = cspdk.cells.mmi2x2_no(width_taper=5.5, length_taper=50, length_mmi=126, width_mmi=12, gap_mmi=0.25, with_bbox=True)
  c.plot()



mmi2x2_rc
----------------------------------------------------

.. autofunction:: cspdk.cells.mmi2x2_rc

.. plot::
  :include-source:

  import cspdk

  c = cspdk.cells.mmi2x2_rc(width_taper=1.5, length_taper=20, length_mmi=44.8, width_mmi=6, gap_mmi=0.53, with_bbox=True)
  c.plot()



mmi2x2_ro
----------------------------------------------------

.. autofunction:: cspdk.cells.mmi2x2_ro

.. plot::
  :include-source:

  import cspdk

  c = cspdk.cells.mmi2x2_ro(width_taper=1.5, length_taper=20, length_mmi=55, width_mmi=6, gap_mmi=0.53, with_bbox=True)
  c.plot()



mmi2x2_sc
----------------------------------------------------

.. autofunction:: cspdk.cells.mmi2x2_sc

.. plot::
  :include-source:

  import cspdk

  c = cspdk.cells.mmi2x2_sc(width_taper=1.5, length_taper=20, length_mmi=42.5, width_mmi=6, gap_mmi=0.5, with_bbox=True)
  c.plot()



mmi2x2_so
----------------------------------------------------

.. autofunction:: cspdk.cells.mmi2x2_so

.. plot::
  :include-source:

  import cspdk

  c = cspdk.cells.mmi2x2_so(width_taper=1.5, length_taper=20, length_mmi=53.5, width_mmi=6, gap_mmi=0.53, with_bbox=True)
  c.plot()



pad
----------------------------------------------------

.. autofunction:: cspdk.cells.pad

.. plot::
  :include-source:

  import cspdk

  c = cspdk.cells.pad(size=(100.0, 100.0), layer=(41, 0), port_inclusion=0)
  c.plot()



straight_nc
----------------------------------------------------

.. autofunction:: cspdk.cells.straight_nc

.. plot::
  :include-source:

  import cspdk

  c = cspdk.cells.straight_nc(length=10.0, npoints=2, add_pins=True)
  c.plot()



straight_no
----------------------------------------------------

.. autofunction:: cspdk.cells.straight_no

.. plot::
  :include-source:

  import cspdk

  c = cspdk.cells.straight_no(length=10.0, npoints=2, add_pins=True)
  c.plot()



straight_rc
----------------------------------------------------

.. autofunction:: cspdk.cells.straight_rc

.. plot::
  :include-source:

  import cspdk

  c = cspdk.cells.straight_rc(length=10.0, npoints=2, add_pins=True)
  c.plot()



straight_ro
----------------------------------------------------

.. autofunction:: cspdk.cells.straight_ro

.. plot::
  :include-source:

  import cspdk

  c = cspdk.cells.straight_ro(length=10.0, npoints=2, add_pins=True)
  c.plot()



straight_sc
----------------------------------------------------

.. autofunction:: cspdk.cells.straight_sc

.. plot::
  :include-source:

  import cspdk

  c = cspdk.cells.straight_sc(length=10.0, npoints=2, add_pins=True)
  c.plot()



straight_so
----------------------------------------------------

.. autofunction:: cspdk.cells.straight_so

.. plot::
  :include-source:

  import cspdk

  c = cspdk.cells.straight_so(length=10.0, npoints=2, add_pins=True)
  c.plot()



trans_sc_rc10
----------------------------------------------------

.. autofunction:: cspdk.cells.trans_sc_rc10

.. plot::
  :include-source:

  import cspdk

  c = cspdk.cells.trans_sc_rc10(length=10, npoints=2, linear=True, width_type='sine')
  c.plot()



trans_sc_rc20
----------------------------------------------------

.. autofunction:: cspdk.cells.trans_sc_rc20

.. plot::
  :include-source:

  import cspdk

  c = cspdk.cells.trans_sc_rc20(length=20, npoints=2, linear=True, width_type='sine')
  c.plot()



trans_sc_rc50
----------------------------------------------------

.. autofunction:: cspdk.cells.trans_sc_rc50

.. plot::
  :include-source:

  import cspdk

  c = cspdk.cells.trans_sc_rc50(length=50, npoints=2, linear=True, width_type='sine')
  c.plot()
