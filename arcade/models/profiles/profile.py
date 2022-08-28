from abc import ABC, abstractmethod

import numpy as np

from arcade.models.types import BaseModel
from arcade.models.base import ModelType


class TypeOfProfileType(BaseModel, ABC):

    @property
    @abstractmethod
    def coords(self) -> np.ndarray:
        pass


class PointListType(TypeOfProfileType):
    pass


class StandardProfileType(TypeOfProfileType):
    pass


class NacaAirfoilType(TypeOfProfileType):
    pass


class ProfileType(ModelType):
    pass


class ProfileGeometryType(ProfileType):
    pass
