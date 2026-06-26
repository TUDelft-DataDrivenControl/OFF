from __future__ import annotations

from abc import ABC, abstractmethod


class TurbineController(ABC):
    """Base interface for local (per-turbine) controller."""
    
    def get_citation(self) -> str:
        """ Returns a citation string for the turbine controller. Default implementation returns a generic citation.

        Returns:
            str: Citation string for the turbine controller.
        """
        return "No specific Turbine Controller citation available."

