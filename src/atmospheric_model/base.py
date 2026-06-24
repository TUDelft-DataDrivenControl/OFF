from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any
import numpy as np

class AtmosphericModel_Base(ABC):
    """Base interface for atmospheric state providers."""

    @abstractmethod
    def step(self, dt: float) -> None:
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError
    
    @abstractmethod
    def get_u_mps(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, t: float) -> np.ndarray:
        raise NotImplementedError
    
    @abstractmethod
    def get_v_mps(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, t: float) -> np.ndarray:
        raise NotImplementedError
    
    def get_w_mps(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, t: float) -> np.ndarray:
        """
        Default implementation of get_w() returns zero vertical velocity.
        Override this method in derived classes if vertical velocity is non-negligible.
        """
        assert x.shape == y.shape == z.shape, "x, y, z must have the same shape"
        return np.zeros_like(x)
    
    @abstractmethod
    def get_horizontal_wind_dir_deg(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, t: float) -> np.ndarray:
        raise NotImplementedError