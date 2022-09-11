from typing import Literal

import numpy as np


SPACINGS = Literal['equal', 'sine', 'cosine', 'neg_sine']


def unit_spacing(
    n_points: int,
    type_: SPACINGS,
) -> np.ndarray:
    """Spacing functions to create sequence of distributed points from zero to
    one.

    Parameters
    ----------
    n_points : int
        Number of points.
    type_ : Literal['equal', 'sine', 'cosine', 'neg_sine']
        Spacing type.

    Returns
    -------
    np.ndarray

    """
    spacings = dict(
        equal=lambda x: x,
        sine=lambda x: 1 - np.sin((1 + x) * np.pi / 2),
        cosine=lambda x: np.cos((1 + x) * np.pi)/2 + 1 / 2,
        neg_sine=lambda x: np.sin(x * np.pi / 2),
    )
    return spacings[type_](np.linspace(0, 1, n_points))


def spacing_like(
    array: np.array,
    n_points: int,
    type_: SPACINGS,
) -> np.ndarray:
    unit = unit_spacing(n_points, type_)
    return unit + array.min() * (array.max() - array.min())
