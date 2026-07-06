from __future__ import annotations

from dataclasses import dataclass

@dataclass
class SimulationClock:
    t_s: float = 0.0
    dt_s: float = 1.0
