# TODO: `src/` — Orchestrator & Entry Point

## Overview
`main.py` is the top-level entry point and the central orchestrator. In the existing code this role is split between `off.py` (simulation loop) and `off_interface.py` (user-facing API / facade). The refactor must cleanly separate orchestration from API concerns and satisfy all requirements from the specs doc.

---

## Must Have

- [ ] **Simulation loop** (`Orchestrator` class or equivalent in `main.py`)
  - Implement the time-stepped simulation loop with clearly separated stages:
    `PREDICT → CORRECT → PROPAGATE → CONTROL → STORE`
  - Provide and advance a **shared simulation time** visible to all modules.
  - Enforce **causality**: no module reads a future state of another module.
  - Support **configurable time steps per module** (e.g., wake at 1 s, turbine ODE at 0.1 s).

- [ ] **Single-file / single-point-of-entry configuration**
  - Accept a YAML (or equivalent) configuration file as the sole required input.
  - Programmatic initialisation from Python dicts must also work (for API use).

- [ ] **Public API** (`OFFSimulation` or similar)
  - `create()` / `run()` / `stop()` / `reset()` methods.
  - `get_state()` / `set_state()` for checkpoint / restart support.
  - `export()` supporting at minimum CSV and HDF5 output formats.

- [ ] **Module wiring**
  - Wire `AtmosphericModel → WakeSolver`, `WakeSolver → AtmosphericModel + TurbineModel`,
    `TurbineModel → WakeModel + Controller` according to the interface spec.
  - Allow **signal overrides** at any module boundary (for fault / cyberattack simulation).

- [ ] **Typed, SI-annotated interfaces**
  - All values exchanged between modules must be typed (use `dataclasses` or `TypedDict`).
  - Annotate units in docstrings or via a unit-annotation helper.

- [ ] **`pip install`-able package**
  - Ensure `src/` is structured as a proper Python package with `__init__.py`.
  - Add `pyproject.toml` / `setup.cfg` at repo root if not already present.

---

## Should Have

- [ ] **Pre-run configuration check**
  - Validate module compatibility (e.g., turbine type vs. wake model) before starting the loop.
  - Emit structured warnings / errors with clear messages.

- [ ] **Structured logging**
  - Per-module log-level configuration (debug / info / warning).
  - File + console handlers; include run-ID in log file name.

- [ ] **Multiple simulation cases**
  - Support running a list of configurations (batch mode).

- [ ] **Multiprocessing support**
  - Parallelise independent turbine evaluations or batch cases where suitable.

- [ ] **Output management**
  - Timestamped run directory created automatically.
  - `store_measurements()`, `store_applied_control()`, `store_run_file()` equivalents.
  - `move_output_to()` helper.

---

## Could Have

- [ ] **Module reset** — allow re-initialisation without full re-instantiation.
- [ ] **Online monitoring** — stream intermediate time-series during simulation.
- [ ] **Real-time mode** — 1 s simulation time = 1 s wall time.
- [ ] **Attribution output** — generate `.bib` based on selected model composition.

---

## Migration Notes (from `03_Code/`)

| Existing | New location |
|---|---|
| `off.py::OFF` | `src/main.py` (orchestrator) |
| `off_interface.py::OFFInterface` | `src/main.py` or `src/api.py` |
| `off.py::__dir_init__`, `__logger_init__` | Utility helpers in `src/utils/` |
| `off_interface.py::_gen_FLORIS_yaml` | Adapter in `src/wake_model/` |
