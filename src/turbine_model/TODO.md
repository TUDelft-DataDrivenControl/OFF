# TODO: `src/turbine_model/` — Turbine Model

## Overview
Manages turbine state (axial induction, yaw, tilt, added TI), geometry (rotor position, nacelle), and aerodynamic calculations (Cp, Ct, power). In the existing code this is split across `turbine.py` (model + states) with a sub-folder for turbine controllers. The refactor separates the aerodynamic/dynamic model from its state history.

---

## Must Have

### `AbstractTurbineModel` — base class

- [ ] Define a **well-documented, typed interface** (SI units):
  - `calc_power(wind_speed, yaw, density) -> float`   — P (W)
  - `get_cp(wind_speed, yaw) -> float`                — Cp (-)
  - `get_ct(wind_speed, yaw) -> float`                — Ct (-)
  - `get_rotor_pos() -> np.ndarray`                   — (3,) world XYZ (m)
  - `set_yaw(yaw_deg) -> None`                        — respects yaw-rate limit
  - `set_tilt(tilt_deg) -> None`
  - `calc_yaw_misalignment(wind_dir_deg) -> float`    — yaw (deg)
  - `reset() -> None`
  - `describe() -> dict`                              — capability self-description

- [ ] Port `HAWT_ADM` (Actuator Disk Model) as the primary concrete implementation.

- [ ] Support **multiple interchangeable implementations**:
  - `HAWT_ADM` — steady-state actuator disk.
  - Placeholder/stub for a dynamic turbine model (ODE-based internal dynamics, e.g., drivetrain).

- [ ] **Internal dynamics** — the turbine model must be capable of simulating internal dynamics (e.g., low-pass filter on effective wind, drivetrain inertia). Design the abstract interface to accommodate this from the start (even if not yet implemented):
  - `step(dt, effective_wind_speed) -> TurbineOutput` — advances internal ODE by dt.

- [ ] **Typed output struct** (`TurbineOutput` dataclass):
  - `power` (W), `thrust` (N), `cp` (-), `ct` (-), `yaw` (deg), `added_ti` (-).

---

### `AbstractTurbineStates` — state history container

- [ ] Port `TurbineStatesFLORIDyn` as primary concrete implementation.

- [ ] Define typed interface:
  - `get_current_yaw() -> float`
  - `get_current_ax_ind() -> float`
  - `get_current_cp() -> float`
  - `get_current_ct() -> float`
  - `set_yaw(yaw_deg) -> None`
  - `set_ax_ind(a) -> None`
  - `get_all_yaw() -> np.ndarray`
  - `get_all_ct() -> np.ndarray`
  - `create_interpolated_state(s1, s2, w) -> AbstractTurbineStates`
  - `iterate() -> None`
  - `reset() -> None`

- [ ] Fix formula inconsistency: existing `get_current_cp()` uses hardcoded exponent 2.2 — make `pP` a configurable parameter loaded from turbine YAML.

---

### `TurbineLibrary` — turbine type registry

- [ ] Create a turbine library / registry that maps turbine type names to parameter dicts.
- [ ] Support loading Cp/Ct curves from:
  - YAML files (existing format).
  - PyWake turbine library (`_populate_pywake_curves` logic, currently in `off_interface.py`).
  - FLORIS turbine YAML.
- [ ] Typed validation of loaded curves (monotonicity, value ranges).

---

## Should Have

- [ ] **Compatibility check** — verify turbine type is compatible with selected wake model (e.g., wake model requires Ct-u curve but turbine only supplies axial-induction formula).

- [ ] **Self-description** (`describe()`):
  - Rotor diameter (m), hub height (m).
  - Power calculation method.
  - Supported control inputs (yaw, pitch, torque …).

- [ ] **Yaw-rate limiter** — correctly enforced in `set_yaw()`, not scattered across controller and turbine.

- [ ] **Structured logging** per class.

- [ ] **Multiple simulation cases** — turbine state must reset cleanly between batch runs.

---

## Could Have

- [ ] **ODE-based drivetrain / pitch model** for full dynamic simulation.
- [ ] **Tilt control** — `calc_tilt()` already exists but is not used; wire it into the interface.
- [ ] **Structural load output** (DEL, fatigue) as optional outputs.
- [ ] **C/C++ back-end** for performance-critical power-curve interpolation.

---

## Migration Notes (from `03_Code/`)

| Existing class / file | New location |
|---|---|
| `turbine.py::Turbine` | `src/turbine_model/base.py` |
| `turbine.py::HAWT_ADM` | `src/turbine_model/hawt_adm.py` |
| `turbine.py::TurbineStates` | `src/turbine_model/states.py` |
| `turbine.py::TurbineStatesFLORIDyn` | `src/turbine_model/states.py` |
| `off_interface.py::_populate_pywake_curves` | `src/turbine_model/library.py` |
| `turbine_controller/` (sub-folder) | `src/turbine_model/turbine_controller/` (local turbine-level controller layer) |
