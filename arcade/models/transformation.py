from typing import ClassVar

import numpy as np
import pytransform3d.rotations as pr
import pytransform3d.transformations as pt

from .types import BaseModel, PointType

class TransformationType(BaseModel):
    PRECISION: ClassVar[int] = 5

    translation: PointType | None = PointType(0.0, 0.0, 0.0)
    rotation: PointType | None = PointType(0.0, 0.0, 0.0)
    scale: PointType | None = PointType(1.0, 1.0, 1.0)

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