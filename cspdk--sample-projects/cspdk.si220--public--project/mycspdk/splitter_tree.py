import gdsfactory as gf


@gf.cell
def splitter_tree(cross_section="strip", **kwargs) -> gf.Component:
    c = gf.c.splitter_tree(cross_section=cross_section, **kwargs)
    return c
