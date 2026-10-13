"""Parameter checks shared by the flavours' cells."""

from __future__ import annotations

# Equality passes, as in the design-rule tables; the tolerance absorbs float noise.
_TOLERANCE = 1e-9


def check_min(name: str, value: float | None, minimum: float, source: str) -> None:
    """Raise ValueError when ``value`` is below a design-rule minimum.

    ``None`` means "use the cell's default geometry" and is not checked.

    Args:
        name: parameter name, for the message.
        value: parameter value in um, or None.
        minimum: the design-rule minimum in um.
        source: where the minimum comes from, e.g. the guideline table.
    """
    if value is not None and value < minimum - _TOLERANCE:
        raise ValueError(
            f"{name}={value} um is below the {minimum} um minimum ({source})"
        )
