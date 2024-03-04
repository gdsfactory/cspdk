from functools import partial

from gplugins.sax import models as sm

nm = 1e-3

straight_sc = partial(sm.straight, neff=2.34, ng=4.2, wl0=1.55)
straight_so = partial(sm.straight, neff=2.34, ng=4.2, wl0=1.31)

gc_rectangular_sc = partial(sm.grating_coupler, loss=6, bandwidth=35 * nm, wl0=1.55)
gc_rectangular_so = partial(sm.grating_coupler, loss=6, bandwidth=35 * nm, wl0=1.31)

bend_sc = partial(sm.bend, loss=0.03)
bend_so = partial(sm.bend, loss=0.03)
bend_rc = partial(sm.bend, loss=0.03)
bend_ro = partial(sm.bend, loss=0.03)
bend_nc = partial(sm.bend, loss=0.03)
bend_no = partial(sm.bend, loss=0.03)


models = dict(
    straight_sc=straight_sc,
    straight_so=straight_so,
    gc_rectangular_sc=gc_rectangular_sc,
    gc_rectangular_so=gc_rectangular_so,
    bend_sc=bend_sc,
    bend_so=bend_so,
    bend_rc=bend_rc,
    bend_ro=bend_ro,
    bend_nc=bend_nc,
    bend_no=bend_no,
)
