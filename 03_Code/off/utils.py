import numpy as np

class OFFTools:
    """
    Class which provides handy methods to run the algorithm
    """

    def __init__(self):
        pass

    def deg2rad(self, deg):
        """
        Function to convert the in LES common degree convention into radians for calculation

        :param deg: LES degrees (270 deg pointing along the x-axis, 190 deg along the y axis)
        :return: radians (0 rad pointing along the x-axis, pi/2 rad along the y axis)
        """
        return np.deg2rad(270 - deg)