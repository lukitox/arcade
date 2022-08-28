from typing import ClassVar, Literal

import numpy as np
import pytransform3d.rotations as pr
import pytransform3d.transformations as pt
from pydantic import Field, validate_arguments

from arcade.models.types import BaseModel, PointType


class TransformationType(BaseModel):
    """Type for describing transformations between coordinate systems.

    Attributes
    ----------
    PRECISION : ClassVar[int]
        Floating point precision of the transformation matrices, by default `5`.
    """
    PRECISION: ClassVar[int] = 5

    translation: PointType | None = Field(
        PointType(0.0, 0.0, 0.0),
        description="Translation.",
    )
    rotation: PointType | None = Field(
        PointType(0.0, 0.0, 0.0),
        description=(
            "The rotation is described in degrees via the three euler angles "
            + "around the axes (x, y, z) which are applied in this order."
        ),
    )
    scale: PointType | None = Field(
        PointType(1.0, 1.0, 1.0),
        description="Scaling of the transformation.",
    )

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
            Shape :math:`(4, 4)`

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
            Shape :math:`(4, 4)`

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
            Shape :math:`(4, 4)`

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
            Shape :math:`(4, 4)`

        """
        return np.matmul(
            self.translation_matrix,
            np.matmul(
                self.rotation_matrix,
                self.scale_matrix
            ),
        )

    @staticmethod
    @validate_arguments
    def mirror_matrix(plane: Literal['xy', 'yz', 'xz']) -> np.ndarray:
        """Returns transformation matrices for symmetric objects.

        .. math::

            A_{xy} = diag(1, 1, -1, 1) \\\\
            A_{yz} = diag(-1, 1, 1, 1) \\\\
            A_{xz} = diag(1, -1, 1, 1)

        Parameters
        ----------
        plane : Literal['xy', 'yz', 'xz']
            The symmetry plane.

        Returns
        -------
        np.ndarray
            Shape :math:`(4, 4)`

        """
        diags = {
            'xy': (1, 1, -1, 1),
            'yz': (-1, 1, 1, 1),
            'xz': (1, -1, 1, 1),
        }

        return np.diag(diags[plane])

    @classmethod
    def _round(cls, arr: np.ndarray) -> np.ndarray:
        return np.round(arr, cls.PRECISION)


def transform_array(A: np.ndarray, coords: np.ndarray) -> np.ndarray:
    """Transforms an array of spatial coordinates :math:`(x, y, z, 1)` using
    the provided transformation matrix.

    .. math::

        C = A \\cdot C'

    Parameters
    ----------
    A : np.ndarray
        Transformation matrix of shape :math:`(4, 4)`.
    coords : np.ndarray
        Array with :math:`k` coordinates of shape :math:`(k, 4)`, so every row
        represents a point.

    Returns
    -------
    np.ndarray
        Shape :math:`(k, 4)`

    """
    return np.apply_along_axis(np.matmul, 1, coords, A)


# Some expressoions that are used occasionally and therfore get an alias
unit_length_x = np.array(((0, 0, 0, 1), (1, 0, 0, 1)))
unit_length_y = np.array(((0, 0, 0, 1), (0, 1, 0, 1)))
unit_length_z = np.array(((0, 0, 0, 1), (0, 0, 1, 1)))
