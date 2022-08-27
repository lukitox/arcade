from enum import Enum
from typing import Literal, NamedTuple

from pydantic import BaseModel as PydanticBaseModel
from pydantic import constr


class BaseModel(PydanticBaseModel):
    """BaseModel configured for this Library."""
    __slots__ = ['__weakref__']  # Needed to allow weakrefs

    class Config:
        use_enum_values = True


UidType = constr(
    strip_whitespace=True,
    to_lower=True,
)
"""Unique ID type: a string with some constrictions."""


class PointType(NamedTuple):
    """3-D point type, inherits from `NamedTuple`, and can therefore either be
    accesed via its attributes (x, y, z) or its indices (0, 1, 2).

    .. code-block:: python

        >>> assert PointType(0.0, 1.0, 2.0) == PointType(x=0.0, y=1.0, z=2.0)
        >>> p = PointType(23.0, 42.0, 66.0)
        >>> p.x
        23.0
        >>> p[0]
        23.0
        >>> p[0] = 7.0  # keep in mind that tuples are immutable.
        TypeError: 'PointType' object does not support item assignment

    """
    x: float
    y: float
    z: float


class SpatialPointType(NamedTuple):
    """3-D point type equipped with a fourth extra value (equal to one), so it
    can be multiplied with 4x4-transformation matrices.

    """
    x: float
    y: float
    z: float
    t: Literal[1] = 1


class SymmetryType(str, Enum):
    """Possible types of symmetry for objects."""
    NONE = "none"
    INHERIT = "inherit"
    XY_PLANE = "xy_plane"
    XZ_PLANE = "xz_plane"
    YZ_PLANE = "yz_plane"
