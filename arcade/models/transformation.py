from typing import ClassVar

import numpy as np
import pytransform3d.rotations as pr
import pytransform3d.transformations as pt

from arcade.models.types import BaseModel, PointType

class TransformationType(BaseModel):
    """Type for describing transformations between coordinate systems.

    Parameters
    ----------
    PRECISION : ClassVar[int]
        Floating point precision of the transformation matrices, by default `5`.
    translation : PointType, optional
        Translation, by default `(0.0, 0.0, 0.0)`.
    rotation : PointType, optional
        Rotation, by default `(0.0, 0.0, 0.0)`. The rotation is described in
        degrees via the three euler angles around the axes (x, y, z) which are
        applied in this order.
    scale : PointType, optional
        Scaling of the transformation, by default `(1.0, 1.0, 1.0)`.
    """
    PRECISION: ClassVar[int] = 5

    translation: PointType | None = PointType(0.0, 0.0, 0.0)
    rotation: PointType | None = PointType(0.0, 0.0, 0.0)
    scale: PointType | None = PointType(1.0, 1.0, 1.0)

    @property
    def rotation_matrix(self) -> np.ndarray:
        """Rotation matrix of shape `(4, 4)`. The rotation matrix is constructed
        with extrinsic euler rotation in (x, y, z)-order.

        .. math::

            \\begin{bmatrix} x \\\\ y \\\\ z \\\\ 1 \\end{bmatrix} & =
            R_z \\cdot R_y \\cdot R_x
            \\cdot
            \\begin{bmatrix} x' \\\\ y' \\\\ z' \\\\ 1 \\end{bmatrix}
            \\\\
            \\begin{bmatrix} x \\\\ y \\\\ z \\\\ 1 \\end{bmatrix} & =
            \\begin{bmatrix}
            cos \\theta & - sin \\theta & 0 & 0 \\\\
            sin \\theta & cos \\theta   & 0 & 0 \\\\
            0           & 0             & 0 & 0 \\\\
            0           & 0             & 0 & 1
            \\end{bmatrix}
            \\cdot
            \\begin{bmatrix}
            cos \\psi   & 0 & sin \\psi & 0 \\\\
            0           & 1 & 0         & 0 \\\\
            - sin \\psi & 0 & cos \\psi & 0 \\\\
            0           & 0 & 0           & 1
            \\end{bmatrix}
            \\cdot
            \\begin{bmatrix}
            1 & 0           & 0           & 0 \\\\
            0 & cos \\phi   & - sin \\phi & 0 \\\\
            0 & sin \\phi   & cos \\phi   & 0 \\\\
            0 & 0           & 0           & 1
            \\end{bmatrix}
            \\cdot
            \\begin{bmatrix} x' \\\\ y' \\\\ z' \\\\ 1 \\end{bmatrix}


        Returns
        -------
        np.ndarray
            Shape `(4, 4)`

        """
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
    def scale_matrix(self) -> np.ndarray:
        """Scale matrix.

        .. math::

            \\begin{bmatrix} x \\\\ y \\\\ z \\\\ 1 \\end{bmatrix} =
            \\begin{bmatrix}
            s_x & 0   & 0   & 0 \\\\
            0   & s_y & 0   & 0 \\\\
            0   & 0   & s_z & 0 \\\\
            0   & 0   & 0   & 1
            \\end{bmatrix}
            \\cdot
            \\begin{bmatrix} x' \\\\ y' \\\\ z' \\\\ 1 \\end{bmatrix}

        Returns
        -------
        np.ndarray
            Shape `(4, 4)`

        """
        return self._round(
            pt.transform_from(
                R=np.diag(self.scale),
                p=np.zeros((3,)),
            )
        )

    @property
    def translation_matrix(self) -> np.ndarray:
        """Translation matrix.

        .. math::

            \\begin{bmatrix} x \\\\ y \\\\ z \\\\ 1 \\end{bmatrix} =
            \\begin{bmatrix}
            1 & 0 & 0 & t_x \\\\
            0 & 1 & 0 & t_y \\\\
            0 & 0 & 1 & t_z \\\\
            0 & 0 & 0 & 1
            \\end{bmatrix}
            \\cdot
            \\begin{bmatrix} x' \\\\ y' \\\\ z' \\\\ 1 \\end{bmatrix}

        Returns
        -------
        np.ndarray
            Shape `(4, 4)`

        """
        return self._round(
            pt.transform_from(
                R=np.eye(3),
                p=self.translation,
            )
        )

    @property
    def matrix(self) -> np.ndarray:
        """Transformation matrix.

        Order of transformations:
            1. scale
            2. rotate
            3. translate

        .. math::

            \\begin{bmatrix} x \\\\ y \\\\ z \\\\ 1 \\end{bmatrix} & =
            R_t \\cdot R_r \\cdot R_s
            \\cdot
            \\begin{bmatrix} x' \\\\ y' \\\\ z' \\\\ 1 \\end{bmatrix}

        Returns
        -------
        np.ndarray
            Shape `(4, 4)`

        """
        return np.matmul(
            self.translation_matrix,
            np.matmul(
                self.rotation_matrix,
                self.scale_matrix
            ),
        )

    def _round(self, arr: np.ndarray) -> np.ndarray:
        return np.round(arr, self.PRECISION)
