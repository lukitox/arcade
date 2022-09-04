"""Foo.

.. mermaid::

    classDiagram
    AbstractModelType <|-- ModelType
    AbstractModelType <|-- OriginModelType
    OriginModelType <|-- SpatialModelType

    class AbstractModelType{
        +UidType uid
        +str name
        +str description
        ~bool STRICT
    }

"""
from abc import ABC
from typing import ClassVar
from warnings import warn
from weakref import ref

import numpy as np
from pydantic import Field, validator
from pytransform3d.transform_manager import TransformManager

import arcade.models.transformation as tm
from arcade.models.transformation import TransformationType
from arcade.models.types import BaseModel, UidType


class AbstractModelType(BaseModel, ABC):
    """Abstract model type."""
    _UID: ClassVar[dict[UidType, ref]] = dict()
    STRICT: ClassVar[bool] = False
    uid: UidType = Field(
        ...,
        description=(
            "Unique identifier. Only lowercase characters, underscores and "
            + "dots allowed."
        ),
    )
    name: str | None = Field(description="The object's name.")
    description: str | None = Field(description="Some description string.")

    def __init__(self, **data) -> None:
        super().__init__(**data)
        self._UID[self.uid] = ref(self)

    @validator('uid')
    def uid_is_unique(cls, uid):
        """Ensures that the entered uid is not already taken."""
        if uid not in cls._UID:  # new object
            return uid
        if cls._UID[uid]() is None:  # key exists, but value is dead weakref
            return uid

        msg = f'UID "{uid}" is already assigned to another object.'
        if cls.STRICT:
            raise ValueError(msg)
        warn(msg + f"\nIgnore this warning if you are just overwriting {uid}.")


class ModelType(AbstractModelType):
    pass


class OriginModelType(AbstractModelType):
    """This is the abstract entry type to an :mod:`arcade`- model. It
    represents the object provididing the global coordinate system. Therefore
    it has no further dependenices on other types.

    Attributes
    ----------
    TM : ClassVar[TransformManager]
        The transform manager of an arcade model. Every child instance of this
        class registers it's own local COS relative so some other COS in the
        transform manager.
        It can then provide transformation matrices between all of the
        registered local coordinate systems.
        Because it's a class variable, it can be accessed from every child
        instance.

    """
    TM: ClassVar[TransformManager] = TransformManager(check=False)


class SpatialModelType(OriginModelType, ModelType):
    """Abstract type for all models from the second hierarchy level on (below
    :class:`OriginModelType`).

    """
    parent: UidType = Field(
        ...,
        description=(
            "Uid of the object this one's coordinate system depends on. The "
            + "corresponding transformation is assigned to the "
            + "`transformation` attribute."
        ),
    )
    transformation: TransformationType | None = Field(
        default=TransformationType(),
        description=(
            "Transform from the parent coordinate system to this object's "
            + "local coordinate system. The default value corresponds to a "
            + "transformation without translation, rotation or scaling."
        )
    )

    def __init__(self, **data) -> None:
        """Links the objects local coordinate system relative to the parent
        object."""
        super().__init__(**data)

        self.TM.add_transform(
            self.parent,
            self.uid,
            self.transformation.matrix
        )

    @validator('parent')
    def uid_exists(cls, uid):
        """Ensures that the uid exists."""
        if uid in cls._UID:
            return uid
        raise ValueError(f'Uid "{uid}" does not exist.')

    def _transform_coords(
        self, coords: np.ndarray, cos_uid: UidType | None = None
    ) -> np.ndarray:
        """Transforms an array with :math:`k` coordinates of shape
        :math:`(k, 4)`, so every row represents a point.

        Parameters
        ----------
        coords : np.ndarray
            Array with :math:`k` coordinates of shape :math:`(k, 4)`, so every
            row represents a point.
        cos_uid : UidType, optional
            If `None` (default), take the local COS, else the referenced one.

        Returns
        -------
        np.ndarray
            Shape :math:`(k, 4)`
        """
        A2B = self.transformation.matrix
        if cos_uid is not None:
            _ = self.uid_exists(cos_uid)  # Validate the uid
            A2B = np.matmul(A2B, self.TM.get_transform(cos_uid, self.uid))

        return tm.transform_array(A2B, coords)
