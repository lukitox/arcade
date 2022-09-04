from itertools import starmap

import numpy as np
from pydantic import Field, conint
from scipy.interpolate import interp1d

from arcade.models.base import ModelType
from arcade.models.profiles.profile import PointListType, TypeOfProfileType
from arcade.models.types import Naca4DigitType, SpatialPointType
from arcade.models.util import SPACINGS, unit_spacing


class AirfoilType(TypeOfProfileType):
    """Aerodynamic airfoil. The airfoil's coordinates are defined in the
    :math:`(x, z)`-plane, ordered from the trailing edge around the top and
    back."""

    @property
    def top(self) -> np.ndarray:
        """The top (suction) side coordinates.

        Returns
        -------
        np.ndarray
            Shape :math:`(4, k)`

        """
        leading_edge = self.coords[0].argmin()
        return self.coords[:, leading_edge:]

    @property
    def bottom(self) -> np.ndarray:
        """The bottom (pressure) side coordinates.

        Returns
        -------
        np.ndarray
            Shape :math:`(4, k)`

        """
        leading_edge = self.coords[0].argmin()
        return self.coords[:, :leading_edge]

    def query(
        self,
        points: np.ndarray,
        kind: str | None = 'cubic',
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get the profile coordinates for desired relative locations.

        Parameters
        ----------
        points : np.ndarray
            Array of relative coordinates (values from zero to one).
        kind : str, optional
            Method of interpolation for `scipy.interpolate.interp1d`,
            by default 'cubic'.

        Returns
        -------
        top : np.ndarray
            Top side coordinates. Shape :math:`(4, k)`
        bottom : np.ndarray
            Bottom side coordinates. Shape :math:`(4, k)`
        """
        top = self._interpolate(self.top, points, kind)
        bot = self._interpolate(self.bottom, points, kind)

        return top, bot

    @staticmethod
    def _interpolate(
        array: np.ndarray,
        points: np.ndarray,
        kind: str,
    ) -> np.ndarray:
        """Interpolate the array's coordinates using scipy's `interp1d`
        function"""
        x, _, z, _ = array
        f = interp1d(x, z, kind=kind, fill_value="extrapolate")

        return np.array(
            (points, np.zeros_like(points), f(points), np.ones_like(points))
        )

    @staticmethod
    def _points(
        n_points: int | None = None,
        spacing: SPACINGS | np.ndarray | None = None,
    ) -> np.ndarray:
        """Validate the array or construct array from spacing type and nuber
        of points"""
        if isinstance(spacing, np.ndarray):
            if spacing.max() <= 1.0 and spacing.min() >= 0.0:
                return spacing
            raise ValueError("Values have to be between zero and one.")
        if isinstance(n_points, int) and isinstance(spacing, str):
            return unit_spacing(n_points, spacing)
        raise ValueError(
            "Either a spacing type and an amount of points or an array have "
            + "to be entered"
        )

    def thickness(
        self,
        n_points: int | None = 50,
        spacing: SPACINGS | np.ndarray | None = 'cosine',
        kind: str = 'cubic',
    ) -> np.ndarray:
        """Returns the half-thickness distribution.

        Parameters
        ----------
        n_points : int | None, optional
            Number of points, by default 50
        spacing : SPACINGS | np.ndarray | None, optional
            Spacing type or array with :math:`x`-positions, by default 'cosine'
        kind : str, optional
            Method of interpolation for `scipy.interpolate.interp1d`,
            by default 'cubic'.

        Returns
        -------
        np.ndarray
            Shape :math:`(4, k)`

        """
        points = self._points(n_points, spacing)
        (x, y, z_t, ones), (_, _, z_b, _) = self.query(points, kind)

        return np.array((x, y, (z_t - z_b) / 2, ones))

    def camberline(
        self,
        n_points: int | None = 50,
        spacing: SPACINGS | np.ndarray | None = 'cosine',
        kind: str = 'cubic',
    ) -> np.ndarray:
        """Returns the camber line of the airfoil.

        Parameters
        ----------
        n_points : int | None, optional
            Number of points, by default 50
        spacing : SPACINGS | np.ndarray | None, optional
            Spacing type or array with :math:`x`-positions, by default 'cosine'
        kind : str, optional
            Method of interpolation for `scipy.interpolate.interp1d`,
            by default 'cubic'.

        Returns
        -------
        np.ndarray
            Shape :math:`(4, k)`

        """
        points = self._points(n_points, spacing)
        (x, y, z_t, ones), (_, _, z_b, _) = self.query(points, kind)

        return np.array((x, y, z_t + z_b, ones))

    def interpolate(
        self,
        n_points: int | None = 100,
        spacing: SPACINGS | np.ndarray | None = 'cosine',
        kind: str = 'cubic',
    ) -> np.ndarray:
        """Interpolate the airfoil coordinates to new (normalized)
        :math:`x`-positions. Either by directly providing the coordinates
        or by giving the normalization type.

        Parameters
        ----------
        n_points : int | None, optional
            Number of points, by default 100
        spacing : SPACINGS | np.ndarray | None, optional
            Spacing type or array with :math:`x`-positions, by default 'cosine'
        kind : str, optional
            Method of interpolation for `scipy.interpolate.interp1d`,
            by default 'cubic'.

        Returns
        -------
        np.ndarray
            Shape :math:`(4, k)`

        """
        if isinstance(n_points, int):
            n_points //= 2
        points = self._points(n_points, spacing)
        top, bot = self.query(points, kind)

        return np.hstack((np.flip(top, axis=1), bot))


class NacaAirfoilType(AirfoilType):
    """NACA 4-Series airfoil type. The airfoil's coordinates are defined in the
    :math:`(x, z)`-plane, ordered from the trailing edge around the top and
    back."""
    number: Naca4DigitType = Field(
        ...,
        description='The NACA 4-digit number.'
    )
    spacing: SPACINGS | None = Field(
        'cosine',
        description="Type of profile coordinate distribution.",
    )
    n_points: conint(ge=10) | None = Field(
        100,
        description="The default number of points."
    )

    @property
    def m(self) -> float:
        """Maximum camber.

        Returns
        -------
        float
        """
        return int(self.number[0]) / 100

    @property
    def p(self) -> float:
        """Maximum camber position.

        Returns
        -------
        float
        """
        return int(self.number[1]) / 10

    @property
    def t(self) -> float:
        """Thickness relative to the cord.

        Returns
        -------
        float
        """
        return int(self.number[2:]) / 100

    def camberline(
        self,
        n_points: int | None = 50,
        spacing: SPACINGS | np.ndarray | None = 'cosine',
        **kwargs,
    ) -> np.ndarray:
        """Returns the camber line of the airfoil.

        Parameters
        ----------
        n_points : int | None, optional
            Number of points, by default 50
        spacing : SPACINGS | np.ndarray | None, optional
            Spacing type or array with :math:`x`-positions, by default 'cosine'

        Returns
        -------
        np.ndarray
            Shape :math:`(4, k)`

        """
        points = self._points(n_points, spacing)

        def camber(x: float) -> tuple[float, float, float, float]:
            m, p = self.m, self.p
            if x < p:
                z = (m / p**2) * x * (2 * p - x)
            else:
                z = (m / (1 - p)**2) * (1 - 2 * p + x * (2 * p - x))

            return (x, 0, z, 1)

        vcamber = np.vectorize(camber)

        return np.array(vcamber(points))

    def thickness(
        self,
        n_points: int | None = 50,
        spacing: SPACINGS | np.ndarray | None = 'cosine',
        **kwargs,
    ) -> np.ndarray:
        """Returns the half-thickness distribution.

        Parameters
        ----------
        n_points : int | None, optional
            Number of points, by default 50
        spacing : SPACINGS | np.ndarray | None, optional
            Spacing type or array with :math:`x`-positions, by default 'cosine'

        Returns
        -------
        np.ndarray
            Shape :math:`(4, k)`

        """
        points = self._points(n_points, spacing)

        def thickness(x: float) -> tuple[float, float, float, float]:
            t = self.t
            a0, a1, a2, a3, a4 = 0.2969, -0.126, -0.3516, 0.2843, -0.1015
            z = (5 * t) * (a0 * x**0.5 + x * (a1 + x * (a2 + x * (a3 + x*a4))))

            return (x, 0, z, 1)

        vthickness = np.vectorize(thickness)

        return np.array(vthickness(points))

    @property
    def coords(self) -> np.ndarray:
        """Returns the airfoil coordinates, ordered from the trailing edge
        around the top and back.

        Returns
        -------
        np.ndarray
            Shape :math:`(4, k)`

        """
        x, y, z_t, ones = self.thickness(self.n_points // 2, self.spacing)
        _, _, z_c, _ = self.camberline(self.n_points // 2, self.spacing)

        top = np.array((x, y, z_t + z_c, ones))
        bot = np.array((x, y, z_c - z_t, ones))

        return np.hstack((np.flip(top, axis=1), bot))


class PointListAirfoilType(AirfoilType, PointListType, ModelType):
    """Airfoil defined by a . The airfoil's coordinates are defined in the
    :math:`(x, z)`-plane, ordered from the trailing edge around the top and
    back."""

    @classmethod
    def from_file(cls, path: str, skiprows: int = 1, **kwargs):
        """Alternative constructor to read airfoil coordinates from file
        in the xfoil format.

        Parameters
        ----------
        path : str
            Path.
        skiprows : int, optional
            Number of rows in the airfoil file to skip, by default 1.
        **kwargs
            Normal arguments to the constructor, like `'uid'`.

        Returns
        -------
        PointListAirfoilType

        """
        x, z = np.loadtxt(path, skiprows=skiprows).T
        y = np.zeros_like(x)
        ones = np.ones_like(x)
        array = np.array((x, y, z, ones)).T

        return cls(point_list=list(starmap(SpatialPointType, array)), **kwargs)
