# Using PyWake's Built-in Turbine Library with OFF

## Overview

OFF has been enhanced to support loading turbine models directly from PyWake's built-in library. This allows you to use pre-defined turbine models like DTU10MW, IEA15MW, V80, etc., without manually defining all turbine parameters in the YAML file.

**PyWake Turbine Structure:** Turbines are stored as classes in submodules:
```python
from py_wake.examples.data.dtu10mw._dtu10mw import DTU10MW
wt = DTU10MW()  # Instantiate the turbine
```

## What Changed

### Code Changes

1. **wake_model.py**: Added new functionality to load turbines from PyWake's library
   - New method: `_load_turbine_from_pywake_library()` - Loads turbines from `py_wake.examples.data.wtg`
   - Modified: `set_wind_farm()` - Now checks if PyWake library should be used
   - New settings: `use_pywake_turbine_library` and `pywake_turbine_name`

### YAML Configuration Changes

The YAML file now supports two modes:

#### **Mode 1: Use PyWake's Turbine Library (NEW)**

Set these flags in the `wake > settings` section:

```yaml
wake:
  settings:
    use_pywake_turbine_library: true
    pywake_turbine_name: "DTU10MW"  # Case-sensitive! Must match PyWake class name
```

The `turbine` section becomes minimal - it only needs a placeholder:

```yaml
turbine:
  v80:  # Must match turbine_type in wind_farm section
    name: Vestas V80 2MW (from PyWake library)
    performance:
      Cp_curve:
        Cp_u_values: []
        Cp_u_wind_speeds: []
      Ct_curve:
        Ct_u_values: []
        Ct_u_wind_speeds: []
    hub_height: 70.0  # Placeholder - will be overwritten
    rotor_diameter: 80.0  # Placeholder - will be overwritten
    shaft_tilt: 5.0
    pP: 1.88
    pT: 1.88
    # ... other minimal parameters
```

OFF will automatically extract the correct hub height, diameter, and power/thrust curves from PyWake.

#### **Mode 2: Use YAML-Defined Turbines (ORIGINAL)**

Set the flag to false or omit it:

```yaml
wake:
  settings:
    use_pywake_turbine_library: false  # or omit this line
```

Then provide full turbine definitions in the `turbine` section with complete Cp/Ct curves.

## Available PyWake Turbines

Common turbines available in PyWake's library (actual names may vary by PyWake version):

- **DTU10MW** - DTU 10 MW Reference Wind Turbine
- **IEA15MW** - IEA Wind 15 MW Reference Turbine (if available)
- **V80** - Vestas V80 2 MW (if available)
- **NREL5MW** - NREL 5 MW Reference Turbine (if available)

**Important:** Turbine names are **case-sensitive** and must match the PyWake class name exactly.

### Finding Available Turbines

Turbines are stored in: `py_wake/examples/data/<turbine_folder>/`

Each turbine follows this pattern:
```python
from py_wake.examples.data.dtu10mw._dtu10mw import DTU10MW  # Module name is lowercase
from py_wake.examples.data.iea15mw._iea15mw import IEA15MW
```

To find available turbines:
1. Check the `py_wake/examples/data/` folder in your PyWake installation
2. Each subfolder contains a turbine class
3. Use the **class name** (not folder name) in your YAML configuration

## How It Works

1. When `use_pywake_turbine_library: true`, OFF loads the specified turbine from PyWake
2. OFF automatically extracts turbine parameters (hub_height, diameter) from the PyWake object
3. OFF updates the `turbine_library` dictionary with this information so the rest of OFF can access it
4. PyWake uses its own turbine model for wake calculations
5. OFF uses the extracted parameters for its internal calculations (hub height, layout, etc.)

## Example: 001_two_turbines_yaw_step_pywake.yaml

This file has been updated to demonstrate using PyWake's DTU10MW turbine:

1. In `wind_farm > farm > turbine_type`: Set to `["dtu10mw", "dtu10mw", ...]` (lowercase)
2. In `wake > settings`: Added `use_pywake_turbine_library: true` and `pywake_turbine_name: "DTU10MW"` (class name)
3. In `turbine`: Simplified the `dtu10mw` section to minimal placeholders

## Troubleshooting

### Error: "Turbine not found in PyWake library"
- Check the turbine name spelling - it's case-sensitive
- OFF will list available turbines in the error message

### Error: "Cannot load turbines from PyWake library"
- Ensure PyWake is properly installed: `pip install py_wake`
- Check that `py_wake.examples.data` is available

### Turbine parameters seem incorrect
- Verify the `pywake_turbine_name` matches a valid PyWake turbine
- Check that `use_pywake_turbine_library: true` is set

### OFF can't find turbine parameters
- Ensure the turbine name in `wind_farm > turbine_type` matches the key in the `turbine` section
- Even with PyWake library mode, you need a minimal turbine entry with the correct name

## Benefits

✅ No need to manually define Cp/Ct curves  
✅ Use validated reference turbines from PyWake  
✅ Easier to switch between different turbine models  
✅ Ensures consistency between OFF and PyWake turbine parameters  
✅ Reduces YAML file size and complexity

## Migration Guide

To convert an existing YAML file to use PyWake's library:

1. Add the two flags to `wake > settings`:
   ```yaml
   use_pywake_turbine_library: true
   pywake_turbine_name: "DTU10MW"  # Case-sensitive class name from PyWake
   ```

2. Replace the full turbine definition with a minimal placeholder (see Mode 1 above)

3. Update `wind_farm > turbine_type` if needed to match your chosen turbine

4. Run your simulation - OFF will handle the rest!
