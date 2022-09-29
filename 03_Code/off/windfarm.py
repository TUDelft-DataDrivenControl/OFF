class WindFarm:
    time_step = -1

    def __init__(self, turbines, time_step):
        """
        Object which hosts the turbine array as well as parameters, constants & variables important to the simulation.

        Parameters
        ----------
        turbines : Turbine object list
            List of turbines in the wind farm
        time_step : float
            Time step in seconds
        """
        self.turbines = turbines
        self.time_step = time_step
