from typing import ClassVar

import numpy as np
import pytransform3d.rotations as pr
import pytransform3d.transformations as pt
from pydantic import BaseModel as PydanticBaseModel
from pydantic import Field, validator
from pytransform3d.transform_manager import TransformManager

from .types import UidType


class BaseModel(PydanticBaseModel):

    class Config:
        use_enum_values = True


class OriginModelType(BaseModel):
    TM: ClassVar[TransformManager] = TransformManager()
    UID: ClassVar[set[UidType]] = set()
    uid: UidType

    @validator('uid')
    def _uid_is_unique(cls, uid):
        if not uid in cls.UID:
            cls.UID.add(uid)

            return uid
        raise ValueError(f'UID "{uid}" is already assigned to another object.')


class TransformationType(BaseModel):
    PRECISION: ClassVar[int] = 5

    translation: tuple[float, float, float] | None = (0.0, 0.0, 0.0)
    rotation: tuple[float, float, float] | None = (0.0, 0.0, 0.0)
    scale: tuple[float, float, float] | None = (1.0, 1.0, 1.0)

    @property
    def rotation_matrix(self) -> np.ndarray:
        R =  pr.active_matrix_from_extrinsic_euler_xyz(
            e=np.deg2rad(self.rotation)
        )
        return self._round(
            pt.transform_from(
                R=R,
                p=np.zeros((3,)),
            )
        )

    @property
    def translation_matrix(self) -> np.ndarray:
        return self._round(
            pt.transform_from(
                R=np.eye(3),
                p=self.translation,
            )
        )

    @property
    def matrix(self) -> np.ndarray:
        return self._scale_matrix(
            pt.concat(
                self.translation_matrix,
                self.rotation_matrix,
            )
        )

    def _round(self, arr: np.ndarray) -> np.ndarray:
        return np.round(arr, self.PRECISION)

    def _scale_matrix(self, A2B: np.ndarray, **kwargs) -> np.ndarray:
        return self._round(
            pt.scale_transform(
                A2B=A2B,
                s_xt=self.scale[0],
                s_yt=self.scale[1],
                s_zt=self.scale[2],
                **kwargs,
            )
        )


class ModelType(OriginModelType):
    parent: UidType
    transformation: TransformationType | None = Field(
        default_factory=TransformationType
    )

    def __init__(self, **data) -> None:
        super().__init__(**data)

        self.TM.add_transform(
            data['parent'],
            data['uid'],
            data['transformation'].matrix
        )

    @validator('parent')
    def _parent_exists(cls, parent):
        if parent in cls.UID:
            return parent
        raise ValueError(f'Parent uid "{parent}" does not exist.')
