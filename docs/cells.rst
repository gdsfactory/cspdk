

Cells Si220
=============================


array
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.array

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.array(component='pad', spacing=(150.0, 150.0), columns=6, rows=1, add_ports=True, centered=False)
  c.plot()



bend_nc
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.bend_nc

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.bend_nc(radius=25.0, angle=90.0)
  c.plot()



bend_no
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.bend_no

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.bend_no(radius=25.0, angle=90.0)
  c.plot()



bend_rc
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.bend_rc

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.bend_rc(radius=25.0, angle=90.0)
  c.plot()



bend_ro
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.bend_ro

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.bend_ro(radius=25.0, angle=90.0)
  c.plot()



bend_s
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.bend_s

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.bend_s(size=(11.0, 1.8), cross_section='xs_sc')
  c.plot()



bend_sc
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.bend_sc

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.bend_sc(radius=10.0, angle=90.0)
  c.plot()



bend_so
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.bend_so

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.bend_so(radius=10.0, angle=90.0)
  c.plot()



coupler_nc
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.coupler_nc

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.coupler_nc(gap=0.4, length=20.0, dx=10.0, dy=4.0)
  c.plot()



coupler_no
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.coupler_no

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.coupler_no(gap=0.4, length=20.0, dx=10.0, dy=4.0)
  c.plot()



coupler_rc
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.coupler_rc

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.coupler_rc(gap=0.236, length=20.0, dx=10.0, dy=4.0)
  c.plot()



coupler_ro
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.coupler_ro

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.coupler_ro(gap=0.236, length=20.0, dx=10.0, dy=4.0)
  c.plot()



coupler_sc
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.coupler_sc

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.coupler_sc(gap=0.236, length=20.0, dx=10.0, dy=4.0)
  c.plot()



coupler_so
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.coupler_so

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.coupler_so(gap=0.236, length=20.0, dx=10.0, dy=4.0)
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



die_nc
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.die_nc

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.die_nc()
  c.plot()



die_no
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.die_no

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.die_no()
  c.plot()



die_rc
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.die_rc

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.die_rc()
  c.plot()



die_ro
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.die_ro

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.die_ro()
  c.plot()



die_sc
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.die_sc

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.die_sc()
  c.plot()



die_so
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.die_so

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.die_so()
  c.plot()



gc_elliptical_sc
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.gc_elliptical_sc

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.gc_elliptical_sc(grating_line_width=0.343, fiber_angle=15, wavelength=1.53)
  c.plot()



gc_elliptical_so
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.gc_elliptical_so

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.gc_elliptical_so(grating_line_width=0.343, fiber_angle=15, wavelength=1.31)
  c.plot()



gc_rectangular_nc
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.gc_rectangular_nc

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.gc_rectangular_nc()
  c.plot()



gc_rectangular_no
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.gc_rectangular_no

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.gc_rectangular_no()
  c.plot()



gc_rectangular_rc
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.gc_rectangular_rc

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.gc_rectangular_rc()
  c.plot()



gc_rectangular_ro
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.gc_rectangular_ro

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.gc_rectangular_ro()
  c.plot()



gc_rectangular_sc
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.gc_rectangular_sc

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.gc_rectangular_sc()
  c.plot()



gc_rectangular_so
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.gc_rectangular_so

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.gc_rectangular_so()
  c.plot()



grating_coupler_array
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.grating_coupler_array

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.grating_coupler_array(pitch=127.0, n=6, port_name='o1', rotation=0.0, with_loopback=False, grating_coupler_spacing=0.0, cross_section='xs_nc')
  c.plot()



heater
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.heater

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.heater()
  c.plot()



mmi1x2_nc
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.mmi1x2_nc

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.mmi1x2_nc()
  c.plot()



mmi1x2_no
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.mmi1x2_no

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.mmi1x2_no()
  c.plot()



mmi1x2_rc
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.mmi1x2_rc

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.mmi1x2_rc()
  c.plot()



mmi1x2_ro
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.mmi1x2_ro

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.mmi1x2_ro()
  c.plot()



mmi1x2_sc
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.mmi1x2_sc

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.mmi1x2_sc()
  c.plot()



mmi1x2_so
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.mmi1x2_so

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.mmi1x2_so()
  c.plot()



mmi2x2_nc
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.mmi2x2_nc

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.mmi2x2_nc()
  c.plot()



mmi2x2_no
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.mmi2x2_no

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.mmi2x2_no()
  c.plot()



mmi2x2_rc
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.mmi2x2_rc

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.mmi2x2_rc()
  c.plot()



mmi2x2_ro
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.mmi2x2_ro

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.mmi2x2_ro()
  c.plot()



mmi2x2_sc
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.mmi2x2_sc

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.mmi2x2_sc()
  c.plot()



mmi2x2_so
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.mmi2x2_so

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.mmi2x2_so()
  c.plot()



mzi_nc
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.mzi_nc

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.mzi_nc(delta_length=10.0, length_y=2.0, length_x=0.1, add_electrical_ports_bot=True)
  c.plot()



mzi_no
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.mzi_no

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.mzi_no(delta_length=10.0, length_y=2.0, length_x=0.1, add_electrical_ports_bot=True)
  c.plot()



mzi_rc
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.mzi_rc

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.mzi_rc(delta_length=10.0, length_y=2.0, length_x=0.1, add_electrical_ports_bot=True)
  c.plot()



mzi_ro
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.mzi_ro

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.mzi_ro(delta_length=10.0, length_y=2.0, length_x=0.1, add_electrical_ports_bot=True)
  c.plot()



mzi_sc
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.mzi_sc

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.mzi_sc(delta_length=10.0, length_y=2.0, length_x=0.1, add_electrical_ports_bot=True)
  c.plot()



mzi_so
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.mzi_so

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.mzi_so(delta_length=10.0, length_y=2.0, length_x=0.1, add_electrical_ports_bot=True)
  c.plot()



pad
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.pad

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.pad(size=(100.0, 100.0), layer=(41, 0), port_inclusion=0.0)
  c.plot()



rectangle
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.rectangle

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.rectangle(size=(4.0, 2.0), layer=(99, 0), centered=False, port_type='electrical', port_orientations=(180.0, 90.0, 0.0, -90.0), round_corners_east_west=False, round_corners_north_south=False)
  c.plot()



straight_nc
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.straight_nc

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.straight_nc(length=10.0)
  c.plot()



straight_no
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.straight_no

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.straight_no(length=10.0)
  c.plot()



straight_rc
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.straight_rc

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.straight_rc(length=10.0)
  c.plot()



straight_ro
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.straight_ro

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.straight_ro(length=10.0)
  c.plot()



straight_sc
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.straight_sc

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.straight_sc(length=10.0)
  c.plot()



straight_so
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.straight_so

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.straight_so(length=10.0)
  c.plot()



taper_nc
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.taper_nc

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.taper_nc(length=10.0, width1=0.5)
  c.plot()



taper_no
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.taper_no

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.taper_no(length=10.0, width1=0.5)
  c.plot()



taper_rc
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.taper_rc

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.taper_rc(length=10.0, width1=0.5)
  c.plot()



taper_ro
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.taper_ro

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.taper_ro(length=10.0, width1=0.5)
  c.plot()



taper_sc
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.taper_sc

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.taper_sc(length=10.0, width1=0.5)
  c.plot()



taper_so
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.taper_so

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.taper_so(length=10.0, width1=0.5)
  c.plot()



trans_sc_rc10
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.trans_sc_rc10

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.trans_sc_rc10()
  c.plot()



trans_sc_rc20
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.trans_sc_rc20

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.trans_sc_rc20()
  c.plot()



trans_sc_rc50
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.trans_sc_rc50

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.trans_sc_rc50()
  c.plot()



wire_corner
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.wire_corner

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.wire_corner(cross_section='xs_metal_routing')
  c.plot()
