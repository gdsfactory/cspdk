"""Every model's parameter names and default values, stored as regression data.

The cells' settings are already stored this way (``test_settings_*.yml``); this does
the same for the models. SAX, circulax and GDSFactory+ call a model by parameter
name and fill unset parameters from its defaults, so adding, removing or renaming a
parameter, or changing a default, changes how circuits behave. Such a change fails
here until the regression file is regenerated (``pytest --force-regen``), so it
shows up in the pull request diff.
"""

from __future__ import annotations

import importlib
import inspect
from typing import Any

import pytest
from test_model_contract import FLAVOURS

_VARIADIC = {
    inspect.Parameter.VAR_POSITIONAL: "<*args>",
    inspect.Parameter.VAR_KEYWORD: "<**kwargs>",
}


def _default(parameter: inspect.Parameter) -> Any:
    if parameter.kind in _VARIADIC:
        return _VARIADIC[parameter.kind]
    return _value(parameter.default)


def _value(value: Any) -> Any:
    if value is inspect.Parameter.empty:
        return "<required>"
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, tuple | list):
        return [_value(v) for v in value]
    return repr(value)


@pytest.mark.parametrize("flavour", FLAVOURS)
def test_model_signatures(flavour: str, data_regression) -> None:
    """Parameter names and defaults of every model of the flavour."""
    pdk = importlib.import_module(f"cspdk.{flavour}").PDK
    data_regression.check(
        {
            name: {
                parameter.name: _default(parameter)
                for parameter in inspect.signature(model).parameters.values()
            }
            for name, model in sorted(pdk.models.items())
        }
    )
