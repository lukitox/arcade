import numpy as np
from pydantic import (Field, NonNegativeFloat, PositiveFloat, confloat,
                      validator)

from arcade.models.profiles.profile import StandardProfileType


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

        Returns
        -------
        np.ndarray
            Shape :math:`(4, k)`

        """
        half_width = 0.5
        height = half_width * self.height_to_width_ratio * 2
        r = self.corner_radius

        a = half_width - r
        b = height - 2 * r
        arc = np.pi * r / 2
        lens = np.cumsum((a, arc, b, arc, a))

        def coords(length: float) -> tuple[float, float, float, float]:
            if length < lens[0]:
                return (0, length, height / 2, 1)
            elif length < lens[1]:
                s = length - lens[0]
                return (0, a + np.sin(s / r) * r, b / 2 + np.cos(s / r) * r, 1)
            elif length < lens[2]:
                s = length - lens[1]
                return (0, half_width, b / 2 - s, 1)
            elif length < lens[3]:
                s = length - lens[2]
                return (
                    0, a + np.cos(s / r) * r, -b / 2 - np.sin(s / r) * r, 1
                )
            elif length <= lens[4]:
                return (0, lens[4] - length, - height / 2, 1)
            raise ValueError('Length too long')

        vcoords = np.vectorize(coords)
        points = np.linspace(0, lens[-1], self.n_points)

        return np.array(vcoords(points))


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
