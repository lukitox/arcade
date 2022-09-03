from abc import ABC, abstractmethod

import numpy as np
from pydantic import Field, conint, conlist

from arcade.models.base import ModelType
from arcade.models.types import BaseModel, SpatialPointType


class TypeOfProfileType(BaseModel, ABC):

    @property
    @abstractmethod
    def coords(self) -> np.ndarray:
        pass


class PointListType(TypeOfProfileType):
    point_list: conlist(SpatialPointType, min_items=2)

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


class ProfileType(ModelType):
    pass


class ProfileGeometryType(ProfileType):
    pass
