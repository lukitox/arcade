from itertools import starmap

import numpy as np
from pydantic import Field, constr
from scipy.interpolate import interp1d

from arcade.models.types import SpatialPointType
from arcade.models.util import unit_spacing, SPACINGS
from arcade.models.types import Naca4DigitType, SpatialPointType, UidType

from arcade.models.profiles.profile import TypeOfProfileType, PointListType

class AirfoilType(TypeOfProfileType):

    @property
    def top(self) -> np.ndarray:
        leading_edge = self.coords[0].argmin()
        return self.coords[:, leading_edge:]

    @property
    def bottom(self) -> np.ndarray:
        leading_edge = self.coords[0].argmin()
        return self.coords[:, :leading_edge]

    def _normalize(
        self,
        n_points: int | None = 50,
        spacing: SPACINGS | None = 'cosine',
        kind: str = 'cubic',
    ) -> np.ndarray:
        top = self._interpolate(self.top, n_points, spacing, kind)
        bot = self._interpolate(self.bottom, n_points, spacing, kind)

        return top, bot

    @staticmethod
    def _interpolate(
        array: np.ndarray, n_points: int, spacing: SPACINGS, kind: str
    ) -> np.ndarray:
        x, _, z, _ = array
        pts = unit_spacing(n_points, spacing)
        f = interp1d(x, z, kind=kind, fill_value="extrapolate")

        return np.array((pts, np.zeros_like(pts), f(pts), np.ones_like(pts)))

    def thickness(
        self,
        n_points: int | None = 50,
        spacing: SPACINGS | None = 'cosine',
        kind: str = 'cubic',
    ) -> np.ndarray:
        (x, y, z_t, ones), (_, _, z_b, _) = self._normalize(
            n_points, spacing, kind
        )

        return np.array((x, y, z_t - z_b, ones))

    def camberline(
        self,
        n_points: int | None = 50,
        spacing: SPACINGS | None = 'cosine',
        kind: str = 'cubic',
    ) -> np.ndarray:
        (x, y, z_t, ones), (_, _, z_b, _) = self._normalize(
            n_points, spacing, kind
        )

        return np.array((x, y, z_t + z_b, ones))


class NacaAirfoilType(AirfoilType):
    """NACA 4-Series airfoil type."""
    number: Naca4DigitType = Field(
        ...,
        description='The NACA 4-digit number.'
    )


class PointListAirfoilType(AirfoilType, PointListType):

    @classmethod
    def from_file(cls, path: str, skiprows: int = 1):
        x, z = np.loadtxt(path, skiprows=skiprows).T
        y = np.zeros_like(x)
        ones = np.ones_like(x)
        array = np.array((x, y, z, ones)).T

        return cls(point_list=list(starmap(SpatialPointType, array)))
