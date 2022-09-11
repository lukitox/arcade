from arcade.models.aircraft.wing import WingType
from arcade.models.base import OriginModelType
from arcade.models.aircraft.engine import EnginePositionType
from arcade.models.aircraft.fuselage import FuselageType


class AircraftModelType(OriginModelType):
    engines: list[EnginePositionType]
    fuselages: list[FuselageType] | None = []
    wings: list[WingType] | None = []