class WindFarm:
    """
    Object which hosts the turbine array as well as parameters, constants & variables important to the simulation.
    """
    time_step = -1

    def __init__(self, turbines, time_step):
        self.turbines = turbines
        self.time_step = time_step
