"""cspdk si220.

Known limitation: cspdk.si220.cband and cspdk.si220.oband define cells with
identical names but band-specific geometry (e.g. strip/rib widths 0.45 vs
0.40 um). Activating both PDKs in one Python process returns the
first-built geometry for cells whose settings match, because the KCLayout
cell cache is keyed by cell name. Use one band per process (tests and CI
already run each PDK in a separate pytest invocation).
"""
