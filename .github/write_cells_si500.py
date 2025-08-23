"""Write docs."""

import inspect

from cspdk.si500 import _cells as cells
from cspdk.si500.config import PATH

filepath = PATH.repo / "docs" / "cells_si500.rst"

skip = {}

skip_plot: tuple[str, ...] = ("",)
skip_settings: tuple[str, ...] = ()


with open(filepath, "w+") as f:
    f.write(
        """

Cells Si SOI 500nm
=============================
"""
    )

    for name in sorted(cells.keys()):
        if name in skip or name.startswith("_"):
            continue
        print(name)
        sig = inspect.signature(cells[name])
        kwargs = ", ".join(
            [
                f"{p}={repr(sig.parameters[p].default)}"
                for p in sig.parameters
                if isinstance(sig.parameters[p].default, int | float | str | tuple)
                and p not in skip_settings
            ]
        )
        if name in skip_plot:
            f.write(
                f"""

{name}
----------------------------------------------------

.. autofunction:: cspdk.si500.cells.{name}

"""
            )
        else:
            f.write(
                f"""

{name}
----------------------------------------------------

.. autofunction:: cspdk.si500.cells.{name}

.. plot::
  :include-source:

  import cspdk

  c = cspdk.si500.cells.{name}({kwargs}).copy()
  c.draw_ports()
  c.plot()

"""
            )
