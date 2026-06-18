# TODO: `src/farm_controller/` — Farm Controller

## Overview
Determines turbine set-points (yaw, pitch, torque) each time step. In the existing code this lives in `controller.py`. The farm-level controller aggregates multiple turbine-level controllers and exposes a single `step()` call to the orchestrator.

---

## Must Have

### `AbstractController` — base class

- [ ] Define a **well-documented, typed interface**:
  - `__call__(turbines, ambient, t) -> ControlOutput`  — compute and apply set-points for all turbines.
  - `get_applied_settings() -> pd.DataFrame`           — log of every decision (for CSV export).
  - `update(t) -> None`                                — advance internal state (called before `__call__`).
  - `reset() -> None`                                  — re-initialise to t = 0.
  - `describe() -> dict`                               — capability self-description.

- [ ] Typed output struct (`ControlOutput` dataclass):
  - Per-turbine: `yaw_setpoint` (deg), `pitch_setpoint` (deg), `torque_setpoint` (Nm).

- [ ] Port all existing implementations:
  - [ ] `IdealGreedyBaselineController` — instantaneous wind-alignment, no lag.
  - [ ] `RealisticGreedyBaselineController` — hysteresis dead-band + yaw-rate limit.
  - [ ] `YawSteeringLUTController` — CSV look-up table (wind direction → yaw offsets).
  - [ ] `YawSteeringPrescribedMotionController` — time-series interpolation from CSV/YAML.
  - [ ] `YawSteeringFilteredPrescribedMotionController` — **currently a stub**: implement or document as "reserved".

- [ ] Support **multiple interchangeable implementations** — any controller that satisfies the abstract interface must be plug-and-play without touching core source code.

- [ ] **Exogenous online signals** — reference power set-point or external wind forecast can be injected at runtime without re-instantiating the controller.

---

### Signal override / fault injection

- [ ] Controller output must be **overridable** per turbine per time step (for fault / cyberattack simulation use case from requirements).
- [ ] Provide an `OverrideController` wrapper that injects a fixed or time-series override on top of any underlying controller.

---

## Should Have

- [ ] **Stochastic seed** — if any controller uses randomness (e.g., exploration noise), accept an external seed.

- [ ] **Compatibility check** — verify controller is compatible with turbine type:
  - Does the turbine support yaw actuation?
  - Does the turbine support pitch actuation?

- [ ] **Self-description** (`describe()`):
  - Required turbine inputs (wind direction, rotor speed, …).
  - Supported actuator types (yaw / pitch / torque).
  - Time-step constraints (minimum update interval).

- [ ] **Structured logging** per class.

- [ ] **Multiple simulation cases** — controller must reset cleanly between batch runs.

---

## Could Have

- [ ] **MPC / optimisation-based controller** stub (plug-in point for research controllers).
- [ ] **Distributed / multi-process controller** — controller lives in a separate process communicating via IPC.
- [ ] **Real-time mode** — controller respects wall-clock timing constraints.

---

## Migration Notes (from `03_Code/`)

| Existing class / file | New location |
|---|---|
| `controller.py::Controller` | `src/farm_controller/base.py` |
| `controller.py::IdealGreedyBaselineController` | `src/farm_controller/greedy.py` |
| `controller.py::RealisticGreedyBaselineController` | `src/farm_controller/greedy.py` |
| `controller.py::YawSteeringLUTController` | `src/farm_controller/lut.py` |
| `controller.py::YawSteeringPrescribedMotionController` | `src/farm_controller/prescribed.py` |
| `controller.py::YawSteeringFilteredPrescribedMotionController` | `src/farm_controller/prescribed.py` |
| `turbine_model/turbine_controller/` | → merge into `src/farm_controller/` |
