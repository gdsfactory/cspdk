

Cells Si SOI 500nm
=============================


array
----------------------------------------------------

.. autofunction:: cspdk.si500.cells.array

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si500.cells.array(component='pad', spacing=(150.0, 150.0), columns=6, rows=1, add_ports=True, centered=False)
  c.plot()



bend_rc
----------------------------------------------------

.. autofunction:: cspdk.si500.cells.bend_rc

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si500.cells.bend_rc(radius=25, angle=90.0)
  c.plot()



bend_s
----------------------------------------------------

.. autofunction:: cspdk.si500.cells.bend_s

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si500.cells.bend_s(size=(11.0, 1.8), cross_section='xs_rc')
  c.plot()



coupler_rc
----------------------------------------------------

.. autofunction:: cspdk.si500.cells.coupler_rc

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si500.cells.coupler_rc(gap=0.236, length=20.0, dx=10.0, dy=4.0)
  c.plot()



die_rc
----------------------------------------------------

.. autofunction:: cspdk.si500.cells.die_rc

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si500.cells.die_rc()
  c.plot()



gc_rectangular_rc
----------------------------------------------------

.. autofunction:: cspdk.si500.cells.gc_rectangular_rc

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si500.cells.gc_rectangular_rc()
  c.plot()



grating_coupler_array
----------------------------------------------------

.. autofunction:: cspdk.si500.cells.grating_coupler_array

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si500.cells.grating_coupler_array(pitch=127.0, n=6, port_name='o1', rotation=0.0, with_loopback=False, cross_section='xs_sc')
  c.plot()



mmi1x2_rc
----------------------------------------------------

.. autofunction:: cspdk.si500.cells.mmi1x2_rc

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si500.cells.mmi1x2_rc()
  c.plot()



mmi2x2_rc
----------------------------------------------------

.. autofunction:: cspdk.si500.cells.mmi2x2_rc

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si500.cells.mmi2x2_rc()
  c.plot()



mzi_rc
----------------------------------------------------

.. autofunction:: cspdk.si500.cells.mzi_rc

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si500.cells.mzi_rc(delta_length=10.0, length_y=2.0, length_x=0.1)
  c.plot()



pad
----------------------------------------------------

.. autofunction:: cspdk.si500.cells.pad

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si500.cells.pad(size=(100.0, 100.0), layer=<LayerMapCornerstone.PAD: 6>, port_inclusion=0.0)
  c.plot()



rectangle
----------------------------------------------------

.. autofunction:: cspdk.si500.cells.rectangle

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si500.cells.rectangle(size=(4.0, 2.0), layer=<LayerMapCornerstone.FLOORPLAN: 2>, centered=False, port_type='electrical', port_orientations=(180.0, 90.0, 0.0, -90.0))
  c.plot()



straight_rc
----------------------------------------------------

.. autofunction:: cspdk.si500.cells.straight_rc

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si500.cells.straight_rc(length=10.0)
  c.plot()



taper_rc
----------------------------------------------------

.. autofunction:: cspdk.si500.cells.taper_rc

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si500.cells.taper_rc(length=10.0, width1=0.5)
  c.plot()



wire_corner
----------------------------------------------------

.. autofunction:: cspdk.si500.cells.wire_corner

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si500.cells.wire_corner(cross_section='metal_routing')
  c.plot()
