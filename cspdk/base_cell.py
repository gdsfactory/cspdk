from collections.abc import Callable
from typing import TypeVar

import gdsfactory as gf

F = TypeVar("F", bound=Callable[..., gf.Component])


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
    func.__name__ = name
    return gf.cell(func)
