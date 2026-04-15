import pkgutil
import py_wake
print('py_wake version:', getattr(py_wake, '__version__', 'unknown'))
mods = [m.name for m in pkgutil.walk_packages(py_wake.__path__)]
print('modules:', sorted(mods))

# Try candidate imports for Points
try:
    from py_wake.utils.grid import Points
    print('Found Points in py_wake.utils.grid')
except Exception as e:
    print('py_wake.utils.grid Points import failed:', repr(e))
try:
    from py_wake.utils.model_utils import Points as MUPoints
    print('Found Points in py_wake.utils.model_utils')
except Exception as e:
    print('py_wake.utils.model_utils Points import failed:', repr(e))

# Try flow_map API
try:
    from py_wake.flow_map import FlowMap
    print('Found FlowMap in py_wake.flow_map')
except Exception as e:
    print('py_wake.flow_map FlowMap import failed:', repr(e))
