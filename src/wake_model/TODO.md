# TODO: `src/wake_model/` — Wake Solver & Wake Models

## Overview
Responsible for computing effective wind speed (and other derived quantities) at rotor planes and observation points. In the existing code this is split across `wake_model.py` (parametric deficit calculation) and `wake_solver.py` (bridge/adapter that manages queries and optional visualisation).

---

## Must Have

### `AbstractWakeModel` — base class for all deficit models

- [ ] Define a **well-documented, typed interface** (SI units):
  - `get_measurements(turbine_id, rotor_points, ambient, turbine_states) -> WakeMeasurement`
    - Returns: effective wind speed [u, v] (m/s), added TI (-), and any model-specific extras.
  - `set_wind_farm(layout, turbine_states, ambient_states) -> None`
  - `reset() -> None`
  - `describe() -> dict` — capability self-description (which quantities it provides).

- [ ] Port `PythonGaussianWake` into this module as a concrete implementation.

- [ ] Define an **adapter interface** for external solvers:
  - `FLORISWakeAdapter` — wraps FLORIS; translates OFF interface ↔ FLORIS API.
  - `PyWakeAdapter` — wraps PyWake; translates OFF interface ↔ PyWake API.
  - Each adapter must handle the `_gen_FLORIS_yaml` / PyWake turbine curve extraction logic currently in `off_interface.py`.

- [ ] Implement **superposition rules** as pluggable strategies:
  - Linear, RSS (root-sum-of-squares), `FLS` (free-stream linear superposition).
  - Select via input file.

- [ ] All implementations must support **steady-state flow models** and be **replaceable** with equivalent implementations (FLORIDyn ↔ FLORIS ↔ PyWake ↔ custom user function).

---

### `AbstractWakeSolver` — orchestration layer

- [ ] Define typed interface:
  - `get_measurements(turbine_id) -> tuple[np.ndarray, pd.DataFrame]`
    — wind speeds at rotor + at OPs + measurement DataFrame.
  - `set_wind_farm(wind_farm) -> None`
  - `reset() -> None`

- [ ] Implement `get_wind_speeds_location(locations) -> np.ndarray` — currently `NotImplemented` in the existing `WakeSolver`; needed for visualisation and user queries.

- [ ] Support **configurable time steps** — wake solver can run at a different cadence than the turbine ODE.

- [ ] Support **signal override** at solver output (inject custom wind speed for fault simulation).

---

### `WindFarm` — layout and dependency container

- [ ] Port `windfarm.py::WindFarm` here (or to a shared `src/` location).
- [ ] Typed interface:
  - `get_layout() -> np.ndarray`         — (n, 4): [x, y, z, D] (m)
  - `get_dependencies() -> np.ndarray`   — (n, n) boolean
  - `add_turbine(turbine) -> None`
  - `remove_turbine(idx) -> None`
  - `get_op_world_coordinates() -> np.ndarray`

---

## Should Have

- [ ] **Compatibility check** between wake model and turbine model (e.g., requires Ct curve).
  - Documented compatibility criteria.
  - Automatic check triggered by pre-run validator in orchestrator.

- [ ] **Self-description** (`describe()`) listing:
  - Required turbine properties (Ct, diameter, hub height …).
  - Required atmospheric inputs.
  - Supported features (yaw deflection, added TI, …).

- [ ] **Structured logging** per class.

- [ ] **Multiple simulation cases** — solver must be stateless enough to reset between batch runs.

---

## Could Have

- [ ] **Visualisation helpers** (port `vis_flow_field.py` and `wake_solver.py` visualization methods):
  - `get_tile_u(grid) -> np.ndarray` — effective wind on 2-D meshgrid.
  - `get_op_mountain_u() -> np.ndarray` — wind profile along OP chain.
  - Move visualisation rendering out of solver; solver only provides data arrays.
- [ ] **C/C++ back-end** for performance-critical Gaussian deficit calculation.

---

## Migration Notes (from `03_Code/`)

| Existing class / file | New location |
|---|---|
| `wake_model.py::WakeModel` | `src/wake_model/base.py` |
| `wake_model.py::PythonGaussianWake` | `src/wake_model/gaussian_wake.py` |
| `wake_solver.py::WakeSolver` | `src/wake_model/solver.py` |
| `windfarm.py::WindFarm` | `src/wake_model/wind_farm.py` (or `src/wind_farm.py`) |
| `vis_flow_field.py` | `src/wake_model/visualisation.py` |
| `off_interface.py::_gen_FLORIS_yaml` | `src/wake_model/adapters/floris_adapter.py` |
| `off_interface.py::_populate_pywake_curves` | `src/wake_model/adapters/pywake_adapter.py` |
