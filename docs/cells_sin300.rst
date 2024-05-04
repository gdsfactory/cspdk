

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



bend_s
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.bend_s

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.bend_s(size=(11.0, 1.8), cross_section='xs_nc')
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



gc_elliptical_nc
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.gc_elliptical_nc

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.gc_elliptical_nc(grating_line_width=0.343, fiber_angle=20, wavelength=1.53, neff=1.6, cross_section='xs_nc')
  c.plot()



gc_elliptical_no
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.gc_elliptical_no

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.gc_elliptical_no(grating_line_width=0.343, fiber_angle=20, wavelength=1.31, neff=1.63, cross_section='xs_no')
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



grating_coupler_array
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.grating_coupler_array

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.grating_coupler_array(pitch=127.0, n=6, port_name='o1', rotation=0.0, with_loopback=False, grating_coupler_spacing=0.0, cross_section='xs_nc')
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



wire_corner
----------------------------------------------------

.. autofunction:: cspdk.si220.cells.wire_corner

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si220.cells.wire_corner(cross_section='xs_metal_routing')
  c.plot()
