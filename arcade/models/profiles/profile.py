from abc import ABC, abstractmethod

import numpy as np
from pydantic import (Field, NonNegativeFloat, confloat, conint, conlist,
                      validator, PositiveFloat)

from arcade.models.base import ModelType
from arcade.models.transformation import TransformationType
from arcade.models.types import BaseModel, SpatialPointType


class TypeOfProfileType(BaseModel, ABC):

    @property
    @abstractmethod
    def coords(self) -> np.ndarray:
        pass


class PointListType(TypeOfProfileType):
    point_list: conlist(SpatialPointType, min_items=2, unique_items=True)

    @property
    def coords(self) -> np.ndarray:
        """Returns an array of the profiles coordinates.

        Returns
        -------
        np.ndarray
            Shape :math:`(4, k)`

        """
        return np.array(self.point_list).T


class StandardProfileType(TypeOfProfileType):
    n_points: conint(ge=10) | None = Field(
        100,
        description="The number of supports the profile is discretized with."
    )


class RectangleProfileType(StandardProfileType):
    """The width of the profile is always 1, since scaling is performed after
    referencing it (e.g., in the fuselage).
    The resulting profile is defined by the following equation:

    The profile is defined in the :math:`(y, z)`-Plane, so :math:`x = const`.

    .. math::

        max(|y| - 0.5 + c, 0)^2 + max(|z| - 0.5r + c, 0)^2 = c^2

    """
    corner_radius: confloat(ge=0, le=0.5) = Field(
        ...,
        description="The corner radius."
    )
    height_to_width_ratio: NonNegativeFloat = Field(
        ...,
        description="The height to width ratio."
    )

    @validator('height_to_width_ratio')
    def height_is_greater_than_double_the_radius(cls, v, values):
        if 2 * values['corner_radius'] > v:
            raise ValueError(
                'The profile is not defined. Make sure height is greater than '
                + 'diameter.'
            )
        return v

    @property
    def coords(self) -> np.ndarray:
        """Returns an array of the profiles coordinates.

        .. note::

            The Number of points may be less than `n_points`, since the profile
            is constructed from three elements (arc, radius, arc), and redundant
            points are removed.

            I'm not quite happy with this implementation, but it's not that
            easy to write a function that returns defined, equal spaced
            points and is easy to understand. So this might change in the
            future.

        Returns
        -------
        np.ndarray
            Shape :math:`(4, k)`

        """
        c = self.height_to_width_ratio
        r = self.corner_radius

        len_top = 0.5 - r
        len_rad = np.pi * r / 2
        len_side = c / 2 - r
        lens = (len_top, len_rad, len_side)
        length = sum(lens)

        n_pts = [int((l * self.n_points) // (2 * length)) for l in lens]

        # one more point to crop out afterwards so no points overlap
        points_top = np.array(
            (
                np.zeros(n_pts[0] + 1),
                np.linspace(0, len_top, n_pts[0] + 1),
                np.ones(n_pts[0] + 1) * c / 2,
                np.ones(n_pts[0] + 1),
            )
        )

        pts_rad = np.linspace(0, np.pi / 2, n_pts[1])
        points_rad = np.array(
            (
                np.zeros(n_pts[1]),
                0.5 - r + np.sin(pts_rad) * r,
                len_side  + np.cos(pts_rad) * r,
                np.ones(n_pts[1]),
            )
        )

        # one more point to crop out afterwards so no points overlap
        points_side = np.array(
            (
                np.zeros(n_pts[2] + 2),
                np.ones(n_pts[2] + 2) * 0.5,
                np.linspace(len_side, 0, n_pts[2] + 2),
                np.ones(n_pts[2] + 2),
            )
        )

        points = np.hstack(
            (
                points_top[:, :-1],
                points_rad,
                points_side[:, 1:],
            )
        )

        return np.vstack(
            (
                points[:, :-1].T,
                np.flip(
                    np.matmul(TransformationType.mirror_matrix('xy'), points).T,
                    0,
                ),
            )
        ).T


class SuperEllipseProfileType(StandardProfileType):
    """A profile based on superellipses is composed of an upper and a lower
    semi-ellipse, which may differ from each other in their parameterization.
    The total width and height of the profile is always 1, since scaling is
    performed after referencing (e.g., in the fuselage).

    The `lower_height_fraction` describes the portion of the lower semi-ellipse
    on the total height.
    The resulting profile is defined by the following set of equations:

    .. math::

        &|2y|^{m_{upper}} + |\\frac{z-z_0}{0.5-z_0}|^{n_{upper}} = 1
        ; z\\ge z_0 \\\\
        &|2y|^{m_{lower}} + |\\frac{z-z_0}{0.5+z_0}|^{n_{lower}} = 1
        ; z< z_0 \\\\
        &\\text{with}\\\\
        &z_0 = \\text{lower_height_fraction} - 0.5

    .. note::

        For exponents that are infinitely large, the superellipse converges to
        a rectangle. However, the value Inf is not a valid entry at this point.
    """
    lower_height_fraction: confloat(gt=0, lt=1) = Field(
        ...,
        description=(
            "Fraction of height of the lower semi-ellipse relative to the "
            + "total height."
        ),
    )
    m_lower: PositiveFloat = Field(
        ...,
        description="Exponent m for lower semi-ellipse."
    )
    m_upper: PositiveFloat = Field(
        ...,
        description="Exponent m for upper semi-ellipse."
    )
    n_lower: PositiveFloat = Field(
        ...,
        description="Exponent n for lower semi-ellipse."
    )
    n_upper: PositiveFloat = Field(
        ...,
        description="Exponent n for upper semi-ellipse."
    )

    @property
    def coords(self) -> np.ndarray:
        """Returns an array of the profiles coordinates.

        Returns
        -------
        np.ndarray
            Shape :math:`(4, k)`

        """
        z0 = self.lower_height_fraction - 0.5
        t = np.linspace(0, np.pi / 2, self.n_points // 2)
        c, s = np.cos(t), np.sin(t)

        def top_side(m: float, n: float, z0: float) -> np.ndarray:
            y = np.abs(c)**(2/m) * np.sign(c) / 2
            z = np.abs(s)**(2/n) * np.sign(s) * (0.5 - z0) + z0
            return y, z

        def bot_side(m: float, n: float, z0: float) -> np.ndarray:
            y = np.abs(c)**(2/m) * np.sign(c) / 2
            z = - np.abs(s)**(2/n) * np.sign(s) * (0.5 + z0) + z0
            return y, z

        def assemble(f: callable, m: float, n: float, z0: float) -> np.ndarray:
            y, z = f(m, n, z0)
            return np.array((np.zeros_like(y), y, z, np.ones_like(y)))

        return np.hstack(
            (
                np.flip(assemble(top_side, self.m_upper, self.n_upper, z0), 1),
                assemble(bot_side, self.m_lower, self.n_lower, z0),
            )
        )

class NacaAirfoilType(TypeOfProfileType):
    pass


class ProfileType(ModelType):
    pass


class ProfileGeometryType(ProfileType):
    pass
