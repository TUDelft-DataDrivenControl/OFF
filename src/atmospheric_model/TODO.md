# TODO: `src/atmospheric_model/` — Atmospheric Model

## Overview
Manages atmospheric conditions (wind speed, wind direction, turbulence intensity) that vary spatially and temporally along observation-point chains. In the existing code this is split across `ambient.py` (state container), `ambient_corrector.py` (filtering / data ingestion), and `observation_points.py` (wake tracers).

---

## Must Have

### `AtmosphericModel` — abstract base class

- [ ] Define a **well-documented interface** with typed inputs/outputs and SI-unit annotations:
  - `get_wind_speed(location) -> np.ndarray`  — [u, v] (m/s)
  - `get_wind_speed_abs(location) -> float`   — |U| (m/s)
  - `get_wind_dir(location) -> float`         — deg (meteorological or LES convention, document it)
  - `get_turbulence_intensity(location) -> float` — TI (-)
  - `get_state_at_turbine(turbine_id) -> AmbientState`
  - `create_interpolated_state(s1, s2, weight) -> AmbientState`
  - `iterate()` — advance internal state by one time step
  - `reset()` — re-initialise to t = 0 without full re-instantiation

- [ ] Support **time-series data retrieval** (return full history, not only current step).

- [ ] Support **multiple interchangeable implementations** — define an `AbstractAtmosphericModel` ABC:
  - `FLORIDynAtmospheric` (port of existing `FLORIDynAmbient`)
  - `UniformSteadyAtmospheric` (static, uniform — simplest baseline)
  - Document which features each implementation provides (capability self-description).

- [ ] Support **exogenous online signals** — wind gust / time-varying inflow can be injected at runtime.

- [ ] **Stochastic mode** accepts an **external seed** for reproducibility.

---

### `ObservationPoints` — abstract base class + implementations

- [ ] Port `FLORIDynOPs4` (4-state) and `FLORIDynOPs6` (6-state) into this module.

- [ ] Define typed interface:
  - `get_world_coord() -> np.ndarray`            — (n, 3) world XYZ (m)
  - `get_vec_op_to_turbine() -> np.ndarray`      — (n, 3) vector in OP frame (m)
  - `init_all_states(turbine, ambient) -> None`  — position chain downstream
  - `propagate_ops(dt) -> None`                  — advance OPs by one time step
  - `reset() -> None`

- [ ] Fix **incomplete implementations** from `FLORIDynOPs4` (propagate_ops edge cases).

- [ ] Add **unit tests** for propagation correctness (straight-line, curved wake path).

---

### `AmbientCorrector` — filtering & data ingestion

- [ ] Port `Consensus` and `ExponentialMovingAverage` filters.

- [ ] Define `AbstractFilter` interface:
  - `update(t: float) -> None`
  - `__call__(raw_state: np.ndarray) -> np.ndarray`

- [ ] Document that filters are applied **sequentially** (decorator chain pattern).

- [ ] Support **per-turbine vs. global** ambient values from YAML input.

- [ ] Make filter chain **configurable from input file** (choose which filters to apply and their order).

- [ ] Allow **signal override** at the corrector output (requirement: fault/cyberattack simulation).

---

## Should Have

- [ ] **Compatibility check** — emit warning if selected atmospheric model is incompatible with selected wake model (e.g., 2-D ambient with 3-D wake model).

- [ ] **Self-description** method (`describe() -> dict`) listing:
  - supported states,
  - time-step constraints,
  - required inputs.

- [ ] **Structured logging** — per-class log-level.

---

## Could Have

- [ ] **`get_wind_speed_at()`** and **`get_wind_direction_at()`** (currently marked TODO in `FLORIDynAmbient`) — implement spatial interpolation.
- [ ] LES input reader (import full-field turbulence data from NetCDF / HDF5).
- [ ] Online streaming of atmospheric state during simulation.

---

## Migration Notes (from `03_Code/`)

| Existing class / file | New location |
|---|---|
| `ambient.py::AmbientStates` | `src/atmospheric_model/base.py` |
| `ambient.py::FLORIDynAmbient` | `src/atmospheric_model/floridyn_ambient.py` |
| `ambient_corrector.py::AmbientCorrector` | `src/atmospheric_model/corrector.py` |
| `ambient_corrector.py::Consensus` | `src/atmospheric_model/filters.py` |
| `ambient_corrector.py::ExponentialMovingAverage` | `src/atmospheric_model/filters.py` |
| `observation_points.py::ObservationPoints` | `src/atmospheric_model/observation_points.py` |
| `observation_points.py::FLORIDynOPs4/6` | `src/atmospheric_model/observation_points.py` |
| `states.py::States` | `src/atmospheric_model/states.py` (or shared `src/utils/`) |
