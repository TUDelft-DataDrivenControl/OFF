import numpy as np

from __future__ import annotations

from abc import ABC, abstractmethod

class TurbineModel(ABC):
    """Base interface for turbine aerodynamics/dynamics."""

    @abstractmethod
    def step(self, dt: float, effective_wind_speed_mps: float) -> TurbineOutput:
        raise NotImplementedError

    @abstractmethod
    def set_yaw(self, yaw_deg: float) -> None:
        raise NotImplementedError
    
    def get_citation(self) -> str:
        """ Returns a citation string for the turbine model. Default implementation returns a generic citation.

        Returns:
            str: Citation string for the turbine model.
        """
        return "No specific Turbine Model citation available."

    """ 
    ---------------------------------------
    Observables 
    --------------------------------------- 
    """

    """ 
    Power 
    --------------------------------------- 
    """
    @abstractmethod
    def obs_generator_power_w(self) -> float:
        """ Abstract method to observe the current generator power output of the turbine.

        Raises:
            NotImplementedError: Abstract Method, must be implemented in derived classes.

        Returns:
            float: Current generator power output of the turbine (W).
        """
        raise NotImplementedError
    
    @abstractmethod
    def obs_rotor_power_w(self) -> float:
        """ Abstract method to observe the current rotor power output of the turbine.

        Raises:
            NotImplementedError: Abstract Method, must be implemented in derived classes.

        Returns:
            float: Current rotor power output of the turbine (W).
        """
        raise NotImplementedError

    """ 
    Thrust 
    --------------------------------------- 
    """
    @abstractmethod
    def obs_thrust_coefficient(self) -> float:
        """ Observes the current thrust coefficient of the turbine.

        Returns:
            float: Current thrust coefficient of the turbine (dimensionless).
        """
        raise NotImplementedError
    
    @abstractmethod
    def obs_thrust_force_n(self) -> float:
        """ Observes the current thrust force of the turbine.

        Returns:
            float: Current thrust force of the turbine (N).
        """
        raise NotImplementedError
    

    """ 
    Torque
    --------------------------------------- 
    """
    @abstractmethod
    def obs_aerodynamic_torque_nm(self) -> float:
        """ Observes the current aerodynamic torque of the turbine.

        Returns:
            float: Current aerodynamic torque of the turbine (Nm).
        """
        raise NotImplementedError
    
    def obs_generator_torque_nm(self) -> float:
        """ Observes the current generator torque of the turbine.

        Returns:
            float: Current generator torque of the turbine (Nm).
        """
        return -self.obs_aerodynamic_torque_nm()

    """ 
    Operating Point 
    --------------------------------------- 
    """
    @abstractmethod
    def obs_rotor_speed_radps(self) -> float:
        """ Observes the current rotor speed of the turbine.

        Returns:
            float: Current rotor speed of the turbine (rad/s).
        """
        raise NotImplementedError
    
    def obs_rotor_speed_rpm(self) -> float:
        """ Observes the current rotor speed of the turbine.

        Returns:
            float: Current rotor speed of the turbine (RPM).
        """
        return self.obs_rotor_speed_radps() * 60.0 / (2.0 * 3.141592653589793)
    
    @abstractmethod
    def obs_collective_pitch_angle_deg(self) -> float:
        """ Observes the current collective pitch angle of the turbine.

        Returns:
            float: Current collective pitch angle of the turbine (degrees).
        """
        raise NotImplementedError
    

    """ 
    Measurements & Sensors 
    --------------------------------------- 
    """
    @abstractmethod
    def obs_measured_rotor_averaged_wind_speed_mps(self) -> float:
        """ Observes the current rotor-averaged wind speed of the turbine.

        Returns:
            float: Current rotor-averaged wind speed of the turbine (m/s).
        """
        raise NotImplementedError
    
    @abstractmethod
    def obs_measured_rotor_averaged_wind_dir_deg(self) -> float:
        """ Observes the current rotor-averaged wind direction of the turbine.

        Returns:
            float: Current rotor-averaged wind direction of the turbine (degrees).
        """
        raise NotImplementedError
    
    @abstractmethod
    def obs_measured_turbulence_intensity_percent(self) -> float:
        """ Observes the current turbulence intensity of the turbine.

        Returns:
            float: Current turbulence intensity of the turbine (%).
        """
        raise NotImplementedError
    
    @abstractmethod
    def obs_measured_nacelle_wind_speed_mps(self) -> float:
        """ Observes the current nacelle-measured wind speed of the turbine.

        Returns:
            float: Current nacelle-measured wind speed of the turbine (m/s).
        """
        raise NotImplementedError
    
    @abstractmethod
    def obs_measured_nacelle_wind_dir_deg(self) -> float:
        """ Observes the current nacelle-measured wind direction of the turbine.

        Returns:
            float: Current nacelle-measured wind direction of the turbine (degrees).
        """
        raise NotImplementedError
    
    @abstractmethod
    def obs_measured_rotor_yaw_orientation_deg(self) -> float:
        """ Observes the current nacelle-measured yaw orientation of the turbine.

        Returns:
            float: Current nacelle-measured yaw orientation of the turbine (degrees).
        """
        raise NotImplementedError
    
    """ 
    Location & Geometry 
    --------------------------------------- 
    """
    @abstractmethod
    def obs_turbine_base_location_m(self) -> np.ndarray:
        """ Observes the location of the turbine.

        Returns:
            np.ndarray: Location of the turbine (x, y, z) in meters.
        """
        raise NotImplementedError

    @abstractmethod
    def obs_rotor_hub_height_m(self) -> float:
        """ Observes the hub height of the turbine.

        Returns:
            float: Hub height of the turbine (m).
        """
        raise NotImplementedError
    
    @abstractmethod
    def obs_rotor_diameter_m(self) -> float:
        """ Observes the rotor diameter of the turbine.

        Returns:
            float: Rotor diameter of the turbine (m).
        """
        raise NotImplementedError

    @abstractmethod
    def obs_rotor_radius_m(self) -> float:
        """ Observes the rotor radius of the turbine.

        Returns:
            float: Rotor radius of the turbine (m).
        """
        return self.obs_rotor_diameter_m() / 2.0
    
    @abstractmethod
    def obs_rotor_tilt_deg(self) -> float:
        """ Observes the rotor tilt of the turbine.

        Returns:
            float: Rotor tilt of the turbine (degrees).
        """
        raise NotImplementedError
    
    @abstractmethod
    def obs_rotor_yaw_orientation_deg(self) -> float:
        """ Observes the current yaw orientation of the turbine.

        Returns:
            float: Current yaw orientation of the turbine (degrees).
        """
        raise NotImplementedError
    
    @abstractmethod
    def obs_rotor_yaw_misalignment_deg(self) -> float:
        """ Observes the current yaw misalignment of the turbine.

        Returns:
            float: Current yaw misalignment of the turbine (degrees).
        """
        raise NotImplementedError


    
    
    