from __future__ import annotations

from functools import partial

from gplugins.sax.models import (
    attenuator,
    bend,
    coupler,
    grating_coupler,
    mmi1x2,
    mmi2x2,
    phase_shifter,
    straight,
)

nm = 1e-3


straight_sc = partial(straight, wl0=1.55, neff=2.4, ng=4.2)
straight_so = partial(straight, wl0=1.31, neff=2.4, ng=4.2)
straight_rc = partial(straight, wl0=1.55, neff=2.4, ng=4.2)
straight_ro = partial(straight, wl0=1.31, neff=2.4, ng=4.2)
straight_nc = partial(straight, wl0=1.55, neff=2.4, ng=4.2)
straight_no = partial(straight, wl0=1.31, neff=2.4, ng=4.2)


gc_rectangular_sc = partial(grating_coupler, loss=6, bandwidth=35 * nm, wl0=1.55)
gc_rectangular_so = partial(grating_coupler, loss=6, bandwidth=35 * nm, wl0=1.31)

bend_sc = partial(bend, loss=0.03)
bend_so = partial(bend, loss=0.03)
bend_rc = partial(bend, loss=0.03)
bend_ro = partial(bend, loss=0.03)
bend_nc = partial(bend, loss=0.03)
bend_no = partial(bend, loss=0.03)


models = dict(
    attenuator=attenuator,
    bend_euler=bend,
    bend_nc=bend_nc,
    bend_no=bend_no,
    bend_rc=bend_rc,
    bend_ro=bend_ro,
    bend_sc=bend_sc,
    bend_so=bend_so,
    coupler=coupler,
    gc_rectangular_sc=gc_rectangular_sc,
    gc_rectangular_so=gc_rectangular_so,
    grating_coupler=grating_coupler,
    mmi1x2=mmi1x2,
    mmi2x2=mmi2x2,
    phase_shifter=phase_shifter,
    straight=straight,
    straight_sc=straight_sc,
    straight_so=straight_so,
    straight_rc=straight_rc,
    straight_ro=straight_ro,
    straight_nc=straight_nc,
    straight_no=straight_no,
    taper=straight,
)


if __name__ == "__main__":
    import gplugins.sax as gs

    gs.plot_model(grating_coupler)
    # gs.plot_model(coupler)
