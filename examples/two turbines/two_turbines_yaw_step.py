# DUMMY EXAMPLE of the code application to get an idea how a user might interact with the code.


# Step 1: Create the necessary instances
from AtmosphericModel import AtmosphericModel_HomogeneousFlow


atmospheric_model = AtmosphericModel_HomogeneousFlow(wind_speed_abs_mps=8.0, wind_dir_deg=270.0, turbulence_intensity_percent=6.0)


# Step 2: Create the Simulation
from OFF import OFFOrchestrator

simulation = OFFOrchestrator(atmospheric_model=atmospheric_model)

# Step 3: Run the simulation
simulation_duration_s = 10.0  # Total simulation time in seconds
time_step_s = 1.0  # Time step for each simulation step in seconds

simulation.run_simulation(simulation_duration_s=simulation_duration_s, time_step_s=time_step_s)

# Step 4: Access the results
results = simulation.get_results()

