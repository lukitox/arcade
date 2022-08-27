from typing import ClassVar

from pydantic import Field, validator
from pytransform3d.transform_manager import TransformManager

from .transformation import TransformationType
from .types import BaseModel, UidType


class OriginModelType(BaseModel):
    TM: ClassVar[TransformManager] = TransformManager()
    UID: ClassVar[set[UidType]] = set()
    uid: UidType

    @validator('uid')
    def _uid_is_unique(cls, uid):
        if not uid in cls.UID:
            cls.UID.add(uid)

            return uid
        raise ValueError(f'UID "{uid}" is already assigned to another object.')


class ModelType(OriginModelType):
    parent: UidType
    transformation: TransformationType | None = Field(
        default_factory=TransformationType
    )

    def __init__(self, **data) -> None:
        super().__init__(**data)

        self.TM.add_transform(
            data['parent'],
            data['uid'],
            data['transformation'].matrix
        )

    @validator('parent')
    def _parent_exists(cls, parent):
        if parent in cls.UID:
            return parent
        raise ValueError(f'Parent uid "{parent}" does not exist.')
