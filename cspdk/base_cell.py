from collections.abc import Callable
from functools import partial
from typing import TypeVar

import gdsfactory as gf
from kfactory import KCell

F = TypeVar("F", bound=Callable[..., gf.Component] | partial[KCell])

# NOTE: in an upcoming version of kfactory cells will have a cell.factory_name
# property we'll have to start using that property everywhere in gdsfactory in
# stead of cell.function_name or cell.__name__ to ensure cells of partials get
# correctly registered. For now this slight hack works as well.


def base_cell(name: str, func: F, /) -> F:
    if not isinstance(name, str):
        raise ValueError(
            "The first argument of `base_cell` should be a string "
            "specifying the name of the function / factory. The given "
            f"argument {name!r} is not a string."
        )
    if not callable(func):
        raise ValueError(
            "The second argument of `base_cell` should be the function "
            f"or partial being decorated. The given argument {func!r} "
            "is not callable."
        )

    # this will ensure the .function_name attribute is correctly set.
    func.__name__ = name  # type: ignore

    # this is for testing purposes only. We want to ensure all partials
    # in the Pdk are partials of a base cell.
    func._base_cell = True  # type: ignore

    return gf.cell(func)  # type: ignore
