import numpy as np
from pydantic import conlist, Field, PositiveFloat

import arcade.models.transformation as tm
from arcade.models.base import ModelType, SpatialModelType
from arcade.models.types import BaseModel, SymmetryType, UidType


class ComponentSegmentType(ModelType):
    pass


class PositioningType(BaseModel):
    from_section_uid: UidType = Field(
        ...,
        description="Reference to starting section of the positioning vector."
    )
    to_section_uid: UidType = Field(
        ...,
        description=(
            "Reference to ending section (section to be positioned) of the "
            + "positioning vector."
        ),
    )
    length: PositiveFloat = Field(
        0.0,
        description=(
            "Distance between inner and outer section (length of the "
            + "positioning vector)."
        ),
    )
    dihedral_angle: PositiveFloat = Field(
        0.0,
        description=(
            "Dihedralangle between inner and outer section. This angle equals "
            + "a positive rotation of the positioing vector around the x-axis "
            + "of the wing coordinate system. The angle is defined in degrees."
        ),
    )
    sweep_angle: PositiveFloat = Field(
        0.0,
        description=(
            "Sweep angle between inner and outer section. This angle equals a "
            + "shearing of the positioing vector along the x-axis of the wing "
            + "coordinate system. The angle is defined in degrees."
        ),
    )


class WingElementType(SpatialModelType):
    airfoil_uid: UidType = Field(
        ...,
        description="Reference to a wing airfoil."
    )

    def edges(self, cos_uid: UidType | None = None) -> np.ndarray:
        """Returns an array of the element's leading and trailing edge
        coordinates.

        .. math::
            \\begin{bmatrix}
            x_{le} & x_{te} \\\\
            y_{le} & y_{te} \\\\
            z_{le} & z_{te} \\\\
            1 & 1
            \\end{bmatrix}

        Parameters
        ----------
        cos_uid : UidType, optional
            If `None` (default), take the local COS, else the referenced one.

        Returns
        -------
        np.ndarray
            Shape :math:`(4, 2)`
        """
        return self._transform_coords(tm.unit_length_x, cos_uid=cos_uid)


class WingSectionType(SpatialModelType):
    elements: conlist(
        WingElementType, min_items=1, unique_items=True,
    ) = Field(
        ...,
        description="The section's wing elements"
    )


class WingSegmentType(SpatialModelType):
    pass


class WingType(SpatialModelType):
    component_segments: list[ComponentSegmentType] | None = []
    positionings: list[PositioningType] | None = []
    sections: list[WingSectionType] | None = []
    segments: list[WingSegmentType] | None = []
    symmetry: SymmetryType | None = Field(
        SymmetryType.XZ_PLANE,
        description="",
    )
