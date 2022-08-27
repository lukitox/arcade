from numpy.typing import ArrayLike

from arcade.models.base import ModelType, SpatialModelType


class ComponentSegmentType(ModelType):
    pass


class PositioningType(ModelType):
    pass


class SectionType(SpatialModelType):
    pass


class SegmentType(SpatialModelType):
    pass


class WingType(SpatialModelType):
    component_segments: list[ComponentSegmentType] | None = []
    positionings: list[PositioningType] | None = []
    sections: list[SectionType] | None = []
    segments: list[SegmentType] | None = []

    def foo(self, a: float) -> float:
        """Some docstring

        Parameters
        ----------
        a : float
            Some super explicit docstring.

        Returns
        -------
        float
        """
