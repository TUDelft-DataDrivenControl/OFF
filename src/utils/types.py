from __future__ import annotations

from dataclasses import dataclass, enum


@dataclass
class SimulationClock:
    t_s: float = 0.0
    dt_s: float = 1.0

class SupportType(enum.Enum):
    """ Enum describing level of support for given component. Used for compatibility checks between components. """
    NOT_SUPPORTED           = 0
    OPTIONALLY_SUPPORTED    = 1 # If requested, additional computation is requried, or the component is indirectly provideed by calculation of another component. 
    FULLY_SUPPORTED         = 2

class RequirementType(enum.Enum):
    """ Enum describing level of requirement for given component. Used for compatibility checks between components. """
    NOT_REQUIRED    = 0
    OPTIONAL        = 1
    REQUIRED        = 2