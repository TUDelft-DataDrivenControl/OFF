# Specifications and requirements document

## Must have

- The simulation software consists of at least four modules: **Atmospheric Model**, **Wake Solver**, **Turbine Model**, and **Controller**.
- A **central orchestrator** provides shared simulation time and ensures causality and consistency across all modules.
- The toolbox is a **time-series simulation framework**: time series in, time series out.
- Simulation setup/configuration is possible from a **single input file / single point of entry**.
- Simulations are fully manageable via an **API** (creation, execution, modification, storage).
- The package is readily installable via **`pip install`**.
- Each module has a **well-documented interface** (inputs/outputs, getters/setters, supported functionality).
- Each module supports **multiple interchangeable implementations** (e.g., static and dynamic), provided interfaces are compatible.
- Module implementations may **omit non-applicable functionality** if not required by the implemented model/algorithm.
- The turbine module can simulate **internal dynamics**.
- Wake and turbine models are modular and **replaceable** with equivalent implementations.
- Dynamic wake models are modular and interchangeable via common interfaces.
- Steady-state flow models are replaceable (e.g., FLORIDyn ↔ FLORIS, PyWake, or custom user function).
- Module interactions expose required functionality:
  - Atmospheric model → wake solver
  - Wake solver → atmospheric model and turbine model
  - Turbine model → wake model and controller
- All modules support efficient **time-series data retrieval**.
- Module input/output interfaces are **typed** and preferably **SI unit-annotated**.
- Different configurable time steps/timescales are supported (e.g., slow wake simulation vs. fast ODE dynamics).
- Signals exchanged between modules can be overridden (within routing or module internals), including fault/cyberattack simulation use cases.
- Users can extend toolbox functionality **without modifying core source code**.
- Each module has an individual **Specifications and requirements document** specification.

## Should have

- Modules are extendable when downstream dependencies require additional functionality.
- Users can implement **custom module implementations**.
- Toolbox supports initialization/configuration both:
  - from a single input file,
  - programmatically from Python.
- Modules can interact with **exogenous online signals** (e.g., controller reference power, atmospheric wind gusts).
- Stochastic components accept an **external seed** for reproducibility.
- A **pre-run simulation configuration check** validates setup consistency and compatibility.
- Structured logging is available (debug/info/warning) with per-module log-level configuration.
- Simulation data export supports standard formats (e.g., **CSV**, **HDF5**).
- Users can parameterize and run **multiple simulation cases**.
- Simulation performance is optimized and supports multiprocessing/parallelism where suitable.
- Compatibility checks are supported (e.g., wake/turbine model matching), including:
  - module capability/self-description ("who are you"),
  - documented and automatic compatibility checks,
  - extensible compatibility criteria.

## Could have

- Input/simulation files can be auto-generated or stitched from fragmented user requirements.
- Modules support reset functionality (avoid full re-instantiation).
- Modules support online monitoring/streaming of time-series results during simulation.
- Modules support cross-application interaction or distributed implementations.
- Real-time simulation mode is supported (1 s simulation time = 1 s wall time).
- Optional dependencies are installable only when needed by selected implementations.
- Auto-documentation is generated from code comments.
- Attribution output is generated (e.g., `.bib` file based on selected model composition).
- Online visualization module is available (public communication, real-time testing, etc.).
- Project includes a recognizable/cool logo.

## Would have

- Performance-critical modules support **C/C++ implementations**.
