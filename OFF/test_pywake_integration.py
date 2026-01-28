"""
Test script for PyWake integration in OFF framework

This script tests that the PyWake wake model can be loaded and used
within the OFF framework.
"""

import sys
import os

# Add the OFF code directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '03_Code'))

import off.off_interface as offi
import off.off as off
import time

def test_pywake_integration():
    """Test that PyWake can be loaded and used"""
    
    print("="*60)
    print("Testing PyWake Integration in OFF Framework")
    print("="*60)
    
    start_time = time.time()
    
    try:
        # Create an interface object
        print("\n1. Creating OFFInterface object...")
        oi = offi.OFFInterface()
        print("   ✓ OFFInterface created successfully")
        
        # Load the PyWake example configuration
        print("\n2. Loading PyWake simulation configuration...")
        config_path = f'{off.OFF_PATH}/02_Examples_and_Cases/02_Example_Cases/001_two_turbines_yaw_step_pywake.yaml'
        print(f"   Config path: {config_path}")
        oi.init_simulation_by_path(config_path)
        print("   ✓ PyWake simulation initialized successfully")
        
        # Check that PyWake model was loaded
        print("\n3. Verifying PyWake model...")
        wake_model_name = oi.off_sim.wake_solver.floris_wake.__class__.__name__
        print(f"   Wake model class: {wake_model_name}")
        if wake_model_name == "PyWakeModel":
            print("   ✓ PyWake model loaded correctly")
        else:
            print(f"   ✗ Expected PyWakeModel, got {wake_model_name}")
            return False
        
        # Run a short simulation (just first few timesteps)
        print("\n4. Running short simulation test (first 10 seconds)...")
        original_time_end = oi.off_sim.settings_sim['time end']
        oi.off_sim.settings_sim['time end'] = 10  # Only run 10 seconds for testing
        
        oi.run_sim()
        
        print("   ✓ Simulation ran successfully")
        
        # Restore original time end
        oi.off_sim.settings_sim['time end'] = original_time_end
        
        # Check that measurements were generated
        print("\n5. Checking generated measurements...")
        if oi.measurements is not None and len(oi.measurements) > 0:
            print(f"   Number of measurement rows: {len(oi.measurements)}")
            print(f"   Measurement columns: {list(oi.measurements.columns)}")
            print("   ✓ Measurements generated successfully")
        else:
            print("   ✗ No measurements generated")
            return False
        
        # Store output
        print("\n6. Storing output files...")
        oi.store_measurements()
        oi.store_applied_control()
        oi.store_run_file()
        print("   ✓ Output files stored")
        
        elapsed_time = time.time() - start_time
        
        print("\n" + "="*60)
        print("PyWake Integration Test PASSED")
        print(f"Test completed in {elapsed_time:.2f} seconds")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        
        elapsed_time = time.time() - start_time
        print("\n" + "="*60)
        print("PyWake Integration Test FAILED")
        print(f"Test failed after {elapsed_time:.2f} seconds")
        print("="*60)
        
        return False


if __name__ == "__main__":
    success = test_pywake_integration()
    sys.exit(0 if success else 1)
