from typing import ClassVar, Literal

import numpy as np
from pydantic import Field, validate_arguments

from arcade.models.types import BaseModel, PointType


class TransformationType(BaseModel):
    """Type for describing transformations between coordinate systems.

    Parameters
    ----------
    PRECISION : ClassVar[int]
        Floating point precision of the transformation matrices, by default `5`
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
        """Rotation matrix of shape `(4, 4)`. The rotation matrix is
        constructed with extrinsic euler rotation in (x, y, z)-order.

        .. math::

            \\begin{bmatrix} x \\\\ y \\\\ z \\\\ 1 \\end{bmatrix} & =
            R_z \\cdot R_y \\cdot R_x
            \\cdot
            \\begin{bmatrix} x' \\\\ y' \\\\ z' \\\\ 1 \\end{bmatrix}

        Returns
        -------
        np.ndarray
            Shape :math:`(4, 4)`

        """
        phi, psi, theta = self.rotation
        r_x, r_y, r_z = R_x(phi), R_y(psi), R_z(theta)

        return self._round(np.matmul(r_z, np.matmul(r_y, r_x)))

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
            np.vstack(
                (np.diag((*self.scale, 1))[:-1, :], np.array((0, 0, 0, 1)))
            )
        ).T

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
            np.vstack((np.eye(3, 4), np.array((*self.translation, 1)))).T
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
        Array with :math:`k` coordinates of shape :math:`(4, k)`, so every
        column represents a point.

    Returns
    -------
    np.ndarray
        Shape :math:`(4, k)`

    """
    return np.matmul(A, coords)


@validate_arguments
def R_x(angle: float) -> np.ndarray:
    """Rotation matrix around the x-axis.

    .. math::

        R_x =
        \\begin{bmatrix}
        1 & 0           & 0           & 0 \\\\
        0 & cos \\phi   & - sin \\phi & 0 \\\\
        0 & sin \\phi   & cos \\phi   & 0 \\\\
        0 & 0           & 0           & 1
        \\end{bmatrix}

    Parameters
    ----------
    angle : float
        Rotation angle :math:`\\phi` in degrees.

    Returns
    -------
    np.ndarray
        Shape :math:`(4, 4)`
    """
    phi = np.radians(angle)

    return np.array((
        (1, 0, 0, 0),
        (0, np.cos(phi), - np.sin(phi), 0),
        (0, np.sin(phi), np.cos(phi), 0),
        (0, 0, 0, 1)
    ))


@validate_arguments
def R_y(angle: float) -> np.ndarray:
    """Rotation matrix around the y-axis.

    .. math::

        R_y =
        \\begin{bmatrix}
        cos \\psi   & 0 & sin \\psi & 0 \\\\
        0           & 1 & 0         & 0 \\\\
        - sin \\psi & 0 & cos \\psi & 0 \\\\
        0           & 0 & 0           & 1
        \\end{bmatrix}

    Parameters
    ----------
    angle : float
        Rotation angle :math:`\\psi` in degrees.

    Returns
    -------
    np.ndarray
        Shape :math:`(4, 4)`
    """
    psi = np.radians(angle)

    return np.array((
        (np.cos(psi), 0, np.sin(psi), 0),
        (0, 1, 0, 0),
        (-np.sin(psi), 0, np.cos(psi), 0),
        (0, 0, 0, 1)
    ))


@validate_arguments
def R_z(angle: float) -> np.ndarray:
    """Rotation matrix around the z-axis.

    .. math::

        R_z =
        \\begin{bmatrix}
        cos \\theta & - sin \\theta & 0 & 0 \\\\
        sin \\theta & cos \\theta   & 0 & 0 \\\\
        0           & 0             & 1 & 0 \\\\
        0           & 0             & 0 & 1
        \\end{bmatrix}

    Parameters
    ----------
    angle : float
        Rotation angle :math:`\\theta` in degrees.

    Returns
    -------
    np.ndarray
        Shape :math:`(4, 4)`
    """
    theta = np.radians(angle)

    return np.array((
        (np.cos(theta), -np.sin(theta), 0, 0),
        (np.sin(theta), np.cos(theta), 0, 0),
        (0, 0, 1, 0),
        (0, 0, 0, 1),
    ))


def _shear_matrix(
    plane: Literal['xy', 'yz', 'xz'],
    a: float | None = 0.0,
    b: float | None = 0.0,
) -> np.ndarray:
    """Returns the shear matrices.

    .. math::

        \\begin{bmatrix} x \\\\ y \\\\ z \\\\ 1 \\end{bmatrix} =
        S_{ab} \\cdot
        \\begin{bmatrix} x' \\\\ y' \\\\ z' \\\\ 1 \\end{bmatrix}

    Parameters
    ----------
    plane : Literal['xy', 'yz', 'xz']
        The symmetry plane.
    a : float
        Shear value of the first axis.
    b : float
        Shear value of the second axis.

    Returns
    -------
    np.ndarray
        Shape :math:`(4, 4)`

    """
    id_a_b = {
        'xy': ((0, 2), (1, 2)),
        'yz': ((1, 0), (2, 0)),
        'xz': ((0, 1), (2, 1)),
    }
    id_a, id_b = id_a_b[plane]

    A = np.eye(4)
    A[id_a], A[id_b] = np.sin(a), np.sin(b)

    return A


@validate_arguments
def S_xy(alpha: float, beta: float) -> np.ndarray:
    """Rotation matrix around the xy-plane.

    .. math::

        S_{xy} =
        \\begin{bmatrix}
        1 & 0 & \\alpha & 0 \\\\
        0 & 1 & \\beta & 0 \\\\
        0 & 0 & 1 & 0 \\\\
        0 & 0 & 0 & 1
        \\end{bmatrix}

    Parameters
    ----------
    alpha : float
        Shear angle :math:`\\alpha` along the x-axis in degrees.
    beta : float
        Shear angle :math:`\\beta` along the y-axis in degrees.


    Returns
    -------
    np.ndarray
        Shape :math:`(4, 4)`
    """
    alpha, beta = np.radians(alpha), np.radians(beta)

    return _shear_matrix('xy', alpha, beta)


@validate_arguments
def S_yz(alpha: float, beta: float) -> np.ndarray:
    """Rotation matrix around the xy-plane.

    .. math::

        S_{yz} =
        \\begin{bmatrix}
        1 & 0 & 0 & 0 \\\\
        \\alpha & 1 & 0 & 0 \\\\
        \\beta & 0 & 1 & 0 \\\\
        0 & 0 & 0 & 1
        \\end{bmatrix}

    Parameters
    ----------
    alpha : float
        Shear angle :math:`\\alpha` along the y-axis in degrees.
    beta : float
        Shear angle :math:`\\beta` along the z-axis in degrees.


    Returns
    -------
    np.ndarray
        Shape :math:`(4, 4)`
    """
    alpha, beta = np.radians(alpha), np.radians(beta)

    return _shear_matrix('yz', alpha, beta)


@validate_arguments
def S_xz(alpha: float, beta: float) -> np.ndarray:
    """Rotation matrix around the xy-plane.

    .. math::

        S_{xz} =
        \\begin{bmatrix}
        1 & \\alpha & 0 & 0 \\\\
        0 & 1 & 0 & 0 \\\\
        0 & \\beta & 1 & 0 \\\\
        0 & 0 & 0 & 1
        \\end{bmatrix}

    Parameters
    ----------
    alpha : float
        Shear angle :math:`\\alpha` along the x-axis in degrees.
    beta : float
        Shear angle :math:`\\beta` along the z-axis in degrees.


    Returns
    -------
    np.ndarray
        Shape :math:`(4, 4)`
    """
    alpha, beta = np.radians(alpha), np.radians(beta)

    return _shear_matrix('xz', alpha, beta)


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


# Some expressoions that are used occasionally and therfore get an alias
unit_length_x = np.array(((0, 0, 0, 1), (1, 0, 0, 1))).T
unit_length_y = np.array(((0, 0, 0, 1), (0, 1, 0, 1))).T
unit_length_z = np.array(((0, 0, 0, 1), (0, 0, 1, 1))).T
