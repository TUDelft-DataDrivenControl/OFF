# TODO: `src/turbine_model/turbine_controller/` — Turbine Controller

## Overview
This component is the **turbine-level controller**.
It optionally receives high-level targets from the farm controller, combines them with local turbine measurements, and returns actuator setpoints such as turbine orientation (yaw), pitch, and torque.

---

## Must Have

### `AbstractTurbineController` — base class

- [ ] Define a **well-documented, typed interface** (SI units):
  - `update_from_farm_command(cmd: FarmCommand | None) -> None`
    - Accept optional high-level farm request (e.g., yaw target, derating request, power cap).
  - `observe(measurement: TurbineMeasurement) -> None`
    - Consume local turbine measurements (wind estimate, rotor speed, power, yaw position, status flags).
  - `compute_setpoints(t: float, dt: float) -> TurbineSetpoints`
    - Return local setpoints for the next control step.
  - `reset() -> None`
  - `describe() -> dict` (capability self-description)

- [ ] Define typed dataclasses:
  - `FarmCommand`:
    - `yaw_target_deg: float | None`
    - `power_reference_w: float | None`
    - `curtailment_factor: float | None`
    - `mode: str | None`
  - `TurbineMeasurement`:
    - `wind_speed_abs_mps: float`
    - `wind_dir_deg: float`
    - `rotor_speed_radps: float`
    - `generator_torque_nm: float`
    - `power_w: float`
    - `yaw_deg: float`
    - `pitch_deg: float`
    - `turbulence_intensity: float`
    - `status_flags: dict[str, bool]`
  - `TurbineSetpoints`:
    - `yaw_setpoint_deg: float | None`
    - `pitch_setpoint_deg: float | None`
    - `torque_setpoint_nm: float | None`
    - `axial_induction_setpoint: float | None`

- [ ] Implement a baseline `GreedyYawTurbineController`:
  - Yaw aligns with local wind direction or with farm command if provided.
  - Respect yaw-rate limit and deadband.
  - Return smooth setpoint trajectories (no large jumps).

- [ ] Implement `PassThroughTurbineController`:
  - Minimal adapter that maps farm-level command directly to turbine setpoints.
  - Useful for integration testing of farm-controller behavior.

- [ ] Local safety and bounds checks:
  - Clamp setpoints to turbine actuator limits.
  - Reject invalid setpoints and fall back to safe defaults.

- [ ] Integrate with turbine model interface:
  - Turbine controller outputs must map directly to supported turbine model inputs
    (yaw / pitch / torque / axial induction depending on selected turbine model).

---

## Should Have

- [ ] Local filtering/state estimation for noisy measurements:
  - Exponential moving average for wind direction and speed.
  - Optional latency compensation.

- [ ] Compatibility checks:
  - Warn if controller requests unsupported actuator (e.g., pitch command for yaw-only model).

- [ ] Structured logging:
  - Log commands received from farm controller and final applied setpoints.

- [ ] Time-step decoupling:
  - Support turbine controller update interval different from farm controller interval.

---

## Could Have

- [ ] Fault-tolerant mode:
  - On sensor fault, degrade to safe fallback controller.
- [ ] Soft constraints / optimization:
  - Local objective balancing load and power.
- [ ] Hardware-in-the-loop adapter:
  - Replace model measurements with real turbine telemetry source.

---

## Interfaces to Other Components

- [ ] **Input from `src/farm_controller/`**:
  - Optional high-level targets (`FarmCommand`) for each turbine.

- [ ] **Input from `src/turbine_model/`**:
  - Real-time local turbine measurements (`TurbineMeasurement`).

- [ ] **Output to turbine model / orchestrator**:
  - `TurbineSetpoints` including orientation setpoint (`yaw_setpoint_deg`) and other actuator requests.

---

## Migration Notes (from `03_Code/`)

| Existing source | New location |
|---|---|
| `controller.py` turbine-specific logic | `src/turbine_model/turbine_controller/` |
| `turbine.py::set_yaw` + yaw-rate handling | coordinated with `src/turbine_model/turbine_controller/` |
| `turbine_model/turbine_controller/` | keep as dedicated local controller package |
