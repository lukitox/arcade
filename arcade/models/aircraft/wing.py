import numpy as np
from pydantic import conlist, Field

import arcade.models.transformation as tm
from arcade.models.base import ModelType, SpatialModelType
from arcade.models.types import BaseModel, SymmetryType, UidType


class ComponentSegmentType(ModelType):
    pass


class PositioningType(BaseModel):
    """The positioning describes an additional translation of sections."""
    from_section_uid: UidType = Field(
        ...,
        description="Reference to starting section of the positioning vector."
    )
    length: float = Field(
        0.0,
        description=(
            "Distance between inner and outer section (length of the "
            + "positioning vector)."
        ),
    )
    dihedral: float = Field(
        0.0,
        description=(
            "Dihedralangle between inner and outer section. This angle equals "
            + "a positive rotation of the positioing vector around the x-axis "
            + "of the wing coordinate system. The angle is defined in degrees."
        ),
    )
    sweep: float = Field(
        0.0,
        description=(
            "Sweep angle between inner and outer section. This angle equals a "
            + "shearing of the positioing vector along the x-axis of the wing "
            + "coordinate system. The angle is defined in degrees."
        ),
    )

    @property
    def rotation_matrix(self) -> np.ndarray:
        """_summary_

        Returns
        -------
        np.ndarray
            Shape :math:`(4, 4)`
        """
        return tm.R_x(angle=self.dihedral)

    @property
    def sweep_matrix(self) -> np.ndarray:
        """The positioning's sweep matrix.

        Returns
        -------
        np.ndarray
            Shape :math:`(4, 4)`
        """
        return tm.S_xz(alpha=self.sweep, beta=0.0)

    @property
    def translation_matrix(self) -> np.ndarray:
        """The positioning's translation matrix, resulting from the 'length'
        attribute.

        Returns
        -------
        np.ndarray
            Shape :math:`(4, 4)`
        """
        return np.vstack((np.eye(3, 4), np.array((0, self.length, 0, 1)))).T


class WingElementType(SpatialModelType):
    """Within elements the airfoils of the wing are defined. Each section can
    have one or more elements. Within each element one airfoil have to be
    defined. If e.g. the wing should have a step at this section, two elements
    can be defined for the two airfoils.

    """
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
    positioning: PositioningType | None = Field(
        None,
        description="Positioning of the wing section."
    )

    def __init__(self, **data) -> None:
        """Links the objects local coordinate system relative to the parent
        object."""
        super().__init__(**data)

        if self.positioning is not None:
            self.TM.add_transform(
                from_frame=self.parent,
                to_frame=self.uid,
                A2B=self._apply_positioning,
            )

    @property
    def _apply_positioning(self) -> np.ndarray:
        # length
        length = np.matmul(
            self.positioning.translation_matrix,
            self.TM.get_transform(
                from_frame=self.parent,
                to_frame=self.uid,
            )
        )

        # sweep
        sweep = np.matmul(
            self.positioning.sweep_matrix,
            self.TM.get_transform(
                from_frame=self.positioning.from_section_uid,
                to_frame=self.uid,
            )
        )

        # dihedral
        dihedral = np.matmul(
            self.positioning.rotation_matrix,
            self.TM.get_transform(
                from_frame=self.parent,
                to_frame=self.uid,
            )
        )

        return np.matmul(dihedral, np.matmul(sweep, length))


class WingSegmentType(SpatialModelType):
    pass


class WingType(SpatialModelType):
    component_segments: list[ComponentSegmentType] | None = []
    sections: list[WingSectionType] | None = []
    segments: list[WingSegmentType] | None = []
    symmetry: SymmetryType | None = Field(
        SymmetryType.XZ_PLANE,
        description="",
    )
