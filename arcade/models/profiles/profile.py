from abc import ABC, abstractmethod

import numpy as np
from pydantic import confloat, conint, conlist, Field, NonNegativeFloat

from arcade.models.base import ModelType
from arcade.models.types import BaseModel, SpatialPointType
from arcade.models.transformation import TransformationType


class TypeOfProfileType(BaseModel, ABC):

    @property
    @abstractmethod
    def coords(self) -> np.ndarray:
        pass


class PointListType(TypeOfProfileType):
    point_list: conlist(SpatialPointType, min_items=2, unique_items=True)


class StandardProfileType(TypeOfProfileType):
    n_points: conint(ge=10) | None = Field(
        100,
        description="The number of supports the profile is discretized with."
    )


class RectangleProfileType(StandardProfileType):
    """The width of the profile is always 1, since scaling is performed after
    referencing it (e.g., in the fuselage).
    The resulting profile is defined by the following equation:

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

    @property
    def coords(self) -> float:
        c = self.height_to_width_ratio
        r = self.corner_radius

        len_top = 0.5 - r
        len_rad = np.pi * r / 2
        len_side = c / 2 - r
        lens = (len_top, len_rad, len_side)
        length = sum(lens)

        n_pts = [int((l * self.n_points) // length) for l in lens]

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
        )





class NacaAirfoilType(TypeOfProfileType):
    pass


class ProfileType(ModelType):
    pass


class ProfileGeometryType(ProfileType):
    pass
