from __future__ import annotations
from abc import ABC, abstractmethod
import inspect
from dataclasses import dataclass, field
from enum import Enum

class CompatibilityLevel(Enum):
    NONE            = 0     # Meaning not required or not supported
    UNKNOWN         = 1     # Compatilibility Level unknown (here defined to be at least somewhat supported)
    OPTIONAL        = 2     # Requires additional computation or is indirectly provided by another component
    FULL            = 3     # Full support

@dataclass(frozen=True)
class OFFCompatibility:
    """
    Class describing the compatibility of an OFF module with other modules. Used for compatibility checks between components.
    """
    module_type: str = "OFFModule"
    requires: dict[str, dict[str, CompatibilityLevel]] = field(default_factory=dict)   # Convention is "module_type.component_name": SupportType
    provides: dict[str, CompatibilityLevel] = field(default_factory=dict)              # Convention is "component_name": SupportType

def compatibility(level: CompatibilityLevel):
    """
    Decorator for assigning compatibility levels to class methods
    """
    def decorator(func):
        func._compatibility_level = level
        return func

    return decorator

class OFFModule(ABC):
    """Base interface for OFF modules."""

    REQUIRES: dict[str, dict[str, CompatibilityLevel]]
    COMPATABILITY: CompatibilityLevel = None
    COMPAT_PREFIX = "obs_"
    MODULE_TYPE = "OFFModule"

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        parent_compat = getattr(cls, "compatibility", None)
        merged_requires = {
            module_type: dict(funcs)
                for module_type, funcs in (parent_compat.requires if parent_compat else {}).items()
        }

        own_requires = cls.__dict__.get("REQUIRES", {})
        for module_type, funcs in own_requires.items():
            component = merged_requires.setdefault(module_type, {})
            for name, level in funcs.items():
                component[name] = level

        # Populate provides dictionary
        provides = {
            name: getattr(member, "_compatibility_level", CompatibilityLevel.UNKNOWN)
            for name, member in inspect.getmembers(cls, predicate=callable)
            if name.startswith(cls.COMPAT_PREFIX)
        }

        cls.compatibility = OFFCompatibility(
            module_type=cls.MODULE_TYPE,
            requires=merged_requires,
            provides=provides,
        )

    @abstractmethod
    def step(self, it: int) -> None:
        """ Advances the module by a given number of iterations.

        Args:
            it (int): Current iteration of the simulation. The current real time since simulation start is it * dt, where dt is the global time step.
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError
    
    def get_citation(self) -> str:
        """ Returns a citation string for the module.

        Default implementation returns example BibTeX entries for software and paper citations.

        Returns:
            str: Citation string for the module.
        """
        return (
            "@software{floridyn_off_2026,\n"
            "  author       = {Becker, Marcus and Lejeune, Maxime and van Straalen, Ivo},\n"
            "  title        = {OFF wind farm simulation toolbox},\n"
            "  year         = {2026},\n"
            "  version      = {1.0.0},\n"
            "  publisher    = {GitHub},\n"
            "  url          = {https://github.com/TUDelft-DataDrivenControl/OFF}\n"
            "}\n\n"
            "@article{becker2025ADynamicModel,\n"
            "  author       = {Becker, Marcus and Lejeune, Maxime and Chatelain, Philippe and Allaerts, Dries and Mudafort, Rafael and van Wingerden, Jan-Willem},\n"
            "  title        = {A dynamic open-source model to investigate wake dynamics in response to wind farm flow control strategies},\n"
            "  journal      = {Wind Energy Science},\n"
            "  year         = {2025},\n"
            "  volume       = {10},\n"
            "  pages        = {1055--1075},\n"
            "  doi          = {10.5194/wes-10-1055-2025}\n"
            "}"
        )
    
    @classmethod
    def describe_compatibility(cls) -> str:
        """Return a nicely formatted compatibility report for the module.

        Returns:
            str: Human-readable compatibility report.
        """
        compat = getattr(cls, "compatibility", None)
        if compat is None:
            report = f"Compatibility report for {cls.__name__}: unavailable"
            print(report)
            return report

        lines = [
            f"Compatibility report for {compat.module_type}",
            "=" * (len("Compatibility report for") + len(compat.module_type) + 1),
            "",
            "Requires:",
        ]

        if compat.requires:
            for module_type, funcs in compat.requires.items():
                for name, level in funcs.items():
                    lines.append(f"  - {module_type}.{name}: {level.name.lower()}")
        else:
            lines.append("  - none")

        lines.extend(["", "Provides:"])
        if compat.provides:
            for name, level in compat.provides.items():
                lines.append(f"  - {name}: {level.name.lower()}")
        else:
            lines.append("  - none")

        report = "\n".join(lines)
        print(report)
        # return report
    
def check_compatibility(modules) -> list[tuple]:
    provided: dict[tuple[str, str], CompatibilityLevel] = {}
    compat_infos: list[tuple[type[OFFModule], OFFCompatibility]] = []

    for module in modules:
        if inspect.isclass(module) and issubclass(module, OFFModule):
            module_cls = module
        elif isinstance(module, OFFModule):
            module_cls = type(module)
        else:
            raise TypeError(f"Expected OFFModule class or instance, got {type(module)!r}")

        compat = getattr(module_cls, "compatibility", None)
        if compat is None:
            continue

        compat_infos.append((module_cls, compat))
        for base in module_cls.__mro__:
            if issubclass(base, OFFModule) and base is not OFFModule:
                for name, level in compat.provides.items():
                    key = (base.__name__, name)
                    current = provided.get(key, CompatibilityLevel.NONE)
                    provided[key] = max(current, level, key=lambda item: item.value)

    unmet = []
    for module_cls, compat in compat_infos:
        for module_type, funcs in compat.requires.items():
            for name, min_level in funcs.items():
                have = provided.get((module_type, name), CompatibilityLevel.NONE)
                if have.value < min_level.value:
                    unmet.append((compat.module_type, module_type, name, min_level, have))
    return unmet

if __name__ == '__main__':
    import numpy as np
    class OFFv1(OFFModule):
        MODULE_TYPE = "OFFv1"
        REQUIRES = {
            "OFFv2" : {
                "obs_uvw_v2": CompatibilityLevel.OPTIONAL
            }
        }

        @compatibility(CompatibilityLevel.OPTIONAL)
        def obs_uvw_v1(self, xyz_m: np.ndarray, t_s: float) -> np.ndarray:
            return np.zeros((3, xyz_m.shape[1]))  # Placeholder implementations
        
    class OFFv2(OFFModule):
        MODULE_TYPE = "OFFv2"
        REQUIRES = {
            "OFFv1" : {
                "obs_uvw_v1": CompatibilityLevel.OPTIONAL
            }
        }

        @compatibility(CompatibilityLevel.FULL)
        def obs_uvw_v2(self, xyz_m: np.ndarray, t_s: float) -> np.ndarray:
            return np.zeros((3, xyz_m.shape[1]))  # Placeholder implementations

    class OFFv1_2(OFFv1):
        REQUIRES = {
            "OFFv2": {
                "obs_uvw_v2": CompatibilityLevel.OPTIONAL
            }
        }

        @compatibility(CompatibilityLevel.OPTIONAL)
        def obs_uvw_v1(self, xyz_m: np.ndarray, t_s: float) -> np.ndarray:
            return np.zeros((3, xyz_m.shape[1]))  # Placeholder implementations

        # TurbineModelStatic.req_check_compatibility(WakeModelPassthrough)
    # print(OFFv1.compatibility)
    # print(OFFv2.compatibility)
    # print(OFFv1_2.compatibility)
    print(check_compatibility([OFFv1, OFFv2]))

    # OFFv1.describe_compatibility()

