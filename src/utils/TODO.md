# TODO: `src/utils/` — Shared Utilities

## Overview
Pure helper functions and shared infrastructure used across all modules. In the existing code this is spread across `utils.py` (math/geometry), `states.py` (base state management), `logger.py`, and parts of `off.py`/`off_interface.py`.

---

## Must Have

### `coordinates.py` — angle and vector conversions

- [ ] Port and document all functions from `utils.py`:
  - `ot_deg2rad(deg) -> float`        — LES degrees (270° = +x) → radians.
  - `ot_uv2deg(u, v) -> float`        — [u, v] → LES degree convention.
  - `ot_abs2uv(abs_speed, dir_deg) -> np.ndarray`   — scalar + direction → [u, v].
  - `ot_uv2abs(u, v) -> float`        — [u, v] → magnitude.
  - `ot_get_orientation(wind_dir, yaw) -> float`
  - `ot_get_yaw(wind_dir, orientation) -> float`

- [ ] Add **unit tests** verifying round-trip conversions (abs2uv → uv2abs, deg2rad → rad→deg).

- [ ] Add type hints and SI-unit annotations to all functions.

---

### `rotor_discretisation.py` — rotor plane sampling

- [ ] Port `ot_isocell(n_points) -> tuple[np.ndarray, np.ndarray]`:
  - Returns rotor-plane sample points and weights.
- [ ] Port `ot_get_closest_point_3d_sorted()` and `ot_get_closest_2_points_3d_sorted()`.
- [ ] Add test asserting weights sum to 1 and points lie within unit circle.

---

### `states.py` — abstract state history base class

- [ ] Port `States` (abstract) with:
  - `iterate_states(new_state) -> None`           — circular shift + insert.
  - `iterate_states_and_keep(new_state) -> None`  — shift without losing first entry.
  - `get_ind_state(i) -> np.ndarray`
  - `set_ind_state(i, val) -> None`
  - `get_all_states() -> np.ndarray`
  - `set_all_states(arr) -> None`
  - `init_all_states(init_val) -> None`
  - `reset() -> None`

- [ ] Make `States` a generic base (not OFF-specific): no wind-farm references in this class.

---

### `io.py` — input / output helpers

- [ ] YAML → dict loading (currently `_run_yaml_to_dict` in `off_interface.py`).
- [ ] CSV export helper (measurements, applied control, run file).
- [ ] HDF5 export helper (requirement: support HDF5 in addition to CSV).
- [ ] Timestamped run-directory creator (currently `__dir_init__` in `off.py`).
- [ ] Run-ID generator (currently `__get_runid__` in `off.py`).

---

### `logging_setup.py` — structured logging

- [ ] Centralised logger factory:
  - `get_logger(module_name, level) -> logging.Logger`
  - Console + file handlers; include run-ID in log file name.
  - Per-module log-level configurable from input file.

---

### `typing.py` — shared type aliases and dataclasses

- [ ] Define shared typed data structures used across modules:
  - `AmbientState` dataclass: `wind_speed` (m/s), `wind_dir` (deg), `ti` (-).
  - `TurbineOutput` dataclass: `power` (W), `thrust` (N), `cp`, `ct`, `yaw` (deg), `added_ti` (-).
  - `ControlOutput` dataclass: per-turbine set-points.
  - `WakeMeasurement` dataclass: effective [u, v] (m/s), added TI (-).

---

## Should Have

- [ ] **Package `__init__.py`** — expose a clean public API from `src/utils/`.
- [ ] **`validate_units.py`** — lightweight runtime assertions for SI-unit bounds (e.g., wind speed ≥ 0, TI ∈ [0, 1]).
- [ ] All utility functions are **pure** (no side effects, no module-level state).

---

## Could Have

- [ ] `attribution.py` — collect `.bib` entries from selected model implementations and emit a combined bibliography.
- [ ] `progress.py` — progress bar helpers (port `_print_progress_bar` from `off.py`).

---

## Migration Notes (from `03_Code/`)

| Existing class / file | New location |
|---|---|
| `utils.py` (all functions) | `src/utils/coordinates.py` + `src/utils/rotor_discretisation.py` |
| `states.py::States` | `src/utils/states.py` |
| `logger.py` | `src/utils/logging_setup.py` |
| `off.py::__dir_init__`, `__get_runid__` | `src/utils/io.py` |
| `off_interface.py::_run_yaml_to_dict` | `src/utils/io.py` |
| `off.py::_print_progress_bar` | `src/utils/progress.py` |
