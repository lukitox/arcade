from abc import ABC
from typing import ClassVar
from weakref import ref

from pydantic import Field, validator
from pytransform3d.transform_manager import TransformManager

from arcade.models.transformation import TransformationType
from arcade.models.types import BaseModel, UidType


class AbstractModelType(BaseModel, ABC):
    """Abstract model type.

    Parameters
    ----------
    UID : ClassVar of type set of UidType
        Set of all given uid's, used to validate that no uid is used more than
        once.
    uid : UidType
        Unique identifier.
    name : str, optional
        The object's name, by default `None`.
    description : str, optional
        Some description string for the object, by default `None`.

    """
    UID: ClassVar[dict[UidType, ref]] = dict()
    uid: UidType
    name: str | None = None
    description: str | None = None

    def __init__(self, **data) -> None:
        super().__init__(**data)
        self.UID[self.uid] = ref(self)

    @validator('uid')
    def uid_is_unique(cls, uid):
        """Ensures that the entered uid is not already taken.

        Raises
        ------
        ValueError
            If the uid is already assigned to another object.
        """
        if not uid in cls.UID:  # new object
            return uid
        if cls.UID[uid]() is None:  # key exists, but value is dead weakref
            return uid
        raise ValueError(f'UID "{uid}" is already assigned to another object.')


class ModelType(AbstractModelType):
   pass


class OriginModelType(AbstractModelType):
    """This is the abstract entry type to an :mod:`arcade`- model. It represents
    the object provididing the global coordinate system. Therefore it has no
    further dependenices on other types.

    Parameters
    ----------
    TM : ClassVar of type TransformManager
        The transform manager of an arcade model. Every child instance of this
        class registers it's own local COS relative so some other COS in the
        transform manager.
        It can then provide transformation matrices between all of the
        registered local coordinate systems.
        Because it's a class variable, it can be accessed from every child
        instance.

    """
    TM: ClassVar[TransformManager] = TransformManager()


class SpatialModelType(OriginModelType, ModelType):
    """Abstract type for all models from the second hierarchy level on (below
    :class:`OriginModelType`).

    Parameters
    ----------
    parent : UidType
        Uid of the object this one's coordinate system depends on. The
        corresponding transformation is assigned to the `transformation`
        attribute.
    transformation : TransformationType, optional
        Transform from the parent coordinate system to this object's local
        coordinate system.
        The default value corresponds to a transformation without translation,
        rotation or scaling.

    """
    parent: UidType
    transformation: TransformationType | None = Field(
        default_factory=TransformationType
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
    def parent_exists(cls, parent):
        """Ensures that the parent object exists.

        Raises
        ------
        ValueError
            If the entered parent uid was not assigned to any object.
        """
        if parent in cls.UID:
            return parent
        raise ValueError(f'Parent uid "{parent}" does not exist.')
