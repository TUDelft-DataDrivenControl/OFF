
from __future__ import annotations
import numpy as np
from abc import ABC, abstractmethod

from off.OFFModule import *

class TurbineModel(OFFModule):
    """Base interface for turbine aerodynamics/dynamics."""
        
    MODULE_TYPE = "TurbineModel"

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
    @compatibility(CompatibilityLevel.NONE)
    def obs_generator_power_w(self, t_s: np.float64) -> float:
        """ Observes the current generator power output of the turbine.

        Args:
            t_s (np.float64): Current simulation time in seconds.

        Raises:
            NotImplementedError: Abstract Method, must be implemented in derived classes.

        Returns:
            float: Current generator power output of the turbine (W).
        """
        raise NotImplementedError
    
    @compatibility(CompatibilityLevel.NONE)
    def obs_aerodynamic_power_w(self, t_s: np.float64) -> float:
        """ Observes the current aerodynamic power of the turbine.
        The aerodynamic power is the power extracted from the wind by the rotor, which is then converted to electrical power by the generator, coupled by a potential gearbox.

        Args:
            t_s (np.float64): Current simulation time in seconds.

        Raises:
            NotImplementedError: Abstract Method, must be implemented in derived classes.

        Returns:
            float: Current aerodynamic power of the turbine (W).
        """
        raise NotImplementedError

    @compatibility(CompatibilityLevel.NONE)
    def obs_available_power_w(self, t_s: np.float64) -> float:
        """ Observes the current available power of the turbine.
        The available power is the power that would be generated if the turbine were operating at its optimal conditions, given the current wind speed and direction.

        Args:
            t_s (np.float64): Current simulation time in seconds.

        Raises:
            NotImplementedError: Abstract Method, must be implemented in derived classes.

        Returns:
            float: Current available power of the turbine (W).
        """
        raise NotImplementedError

    @compatibility(CompatibilityLevel.NONE)    
    def obs_power_coefficient(self, t_s: np.float64) -> float:
        """ Observes the current power coefficient of the turbine.
        The power coefficient is a dimensionless number that represents the efficiency of the turbine in converting the kinetic energy of the wind into electrical energy.

        Args:
            t_s (np.float64): Current simulation time in seconds.

        Returns:
            float: Current power coefficient of the turbine (dimensionless).
            
        Raises:
            NotImplementedError: Abstract Method, must be implemented in derived classes.
        """
        raise NotImplementedError

    @compatibility(CompatibilityLevel.NONE)   
    def obs_power_curve(self, t_s: np.float64) -> np.ndarray:
        """ Observes the current power curve of the turbine.
        The power curve is a function that describes the relationship between the wind speed and the power output of the turbine.

        Args:
            t_s (np.float64): Current simulation time in seconds.
        
        Returns:
            np.ndarray: Current power curve of the turbine (W) as a function of wind speed (m/s).

        Raises:
            NotImplementedError: Abstract Method, must be implemented in derived classes.
        """
        raise NotImplementedError

    """ 
    Thrust 
    --------------------------------------- 
    """
    @abstractmethod
    @compatibility(CompatibilityLevel.NONE)
    def obs_thrust_coefficient(self, t_s: np.float64) -> float:
        """ Observes the current thrust coefficient of the turbine.

        Args:
            t_s (np.float64): Current simulation time in seconds.

        Returns:
            float: Current thrust coefficient of the turbine (dimensionless).
        """
        raise NotImplementedError
    
    @compatibility(CompatibilityLevel.NONE)
    def obs_thrust_force_n(self, t_s: np.float64) -> float:
        """ Observes the current thrust force of the turbine.

        Args:
            t_s (np.float64): Current simulation time in seconds.

        Returns:
            float: Current thrust force of the turbine (N).
        """
        raise NotImplementedError
    
    @compatibility(CompatibilityLevel.NONE)
    def obs_thrust_curve(self, t_s: np.float64) -> np.ndarray:
        """ Observes the current thrust curve of the turbine.
        The thrust curve is a function that describes the relationship between the wind speed and the thrust force of the turbine.

        Args:
            t_s (np.float64): Current simulation time in seconds.

        Returns:
            np.ndarray: Current thrust curve of the turbine (N) as a function of wind speed (m/s).
        """
        raise NotImplementedError

    """ 
    Torque
    --------------------------------------- 
    """
    @compatibility(CompatibilityLevel.NONE)
    def obs_aerodynamic_torque_nm(self, t_s: np.float64) -> float:
        """ Observes the current aerodynamic torque of the turbine.

        Args:
            t_s (np.float64): Current simulation time in seconds.

        Returns:
            float: Current aerodynamic torque of the turbine (Nm).
        """
        raise NotImplementedError
    
    @compatibility(CompatibilityLevel.OPTIONAL)
    def obs_generator_torque_nm(self, t_s: np.float64) -> float:
        """ Observes the current generator torque of the turbine.

        Args:
            t_s (np.float64): Current simulation time in seconds.

        Returns:
            float: Current generator torque of the turbine (Nm).
        """
        return -self.obs_aerodynamic_torque_nm(t_s)

    """ 
    Operating Point 
    --------------------------------------- 
    """
    @compatibility(CompatibilityLevel.NONE)
    def obs_rotor_speed_radps(self, t_s: np.float64) -> float:
        """ Observes the current rotor speed of the turbine.

        Args:
            t_s (np.float64): Current simulation time in seconds.

        Returns:
            float: Current rotor speed of the turbine (rad/s).
        """
        raise NotImplementedError
    
    @compatibility(CompatibilityLevel.OPTIONAL)
    def obs_rotor_speed_rpm(self, t_s: np.float64) -> float:
        """ Observes the current rotor speed of the turbine.

        Args:
            t_s (np.float64): Current simulation time in seconds.

        Returns:
            float: Current rotor speed of the turbine (RPM).
        """
        return self.obs_rotor_speed_radps(t_s) * 60.0 / (2.0 * 3.141592653589793)

    @compatibility(CompatibilityLevel.NONE)
    def obs_collective_pitch_angle_deg(self, t_s: np.float64) -> float:
        """ Observes the current collective pitch angle of the turbine.

        Args:
            t_s (np.float64): Current simulation time in seconds.

        Returns:
            float: Current collective pitch angle of the turbine (degrees).
        """
        raise NotImplementedError
    
    @compatibility(CompatibilityLevel.OPTIONAL)
    def obs_individual_pitch_angles_deg(self, t_s: np.float64) -> np.ndarray:
        """ Observes the current individual pitch angles of the turbine.

        Args:
            t_s (np.float64): Current simulation time in seconds.

        Returns:
            np.ndarray: Current individual pitch angles of the turbine (degrees).
        """
        return np.full(self.obs_num_blades(), self.obs_collective_pitch_angle_deg(t_s)) # TODO

    """ 
    Measurements & Sensors 
    --------------------------------------- 
    """
    @compatibility(CompatibilityLevel.NONE)
    def obs_measured_rotor_averaged_wind_speed_mps(self, t_s: np.float64) -> float:
        """ Observes the current rotor-averaged wind speed of the turbine.

        Args:
            t_s (np.float64): Current simulation time in seconds.

        Returns:
            float: Current rotor-averaged wind speed of the turbine (m/s).
        """
        raise NotImplementedError
    
    @compatibility(CompatibilityLevel.NONE)
    def obs_measured_rotor_averaged_wind_dir_deg(self, t_s: np.float64) -> float:
        """ Observes the current rotor-averaged wind direction of the turbine.

        Args:
            t_s (np.float64): Current simulation time in seconds.

        Returns:
            float: Current rotor-averaged wind direction of the turbine (degrees).
        """
        raise NotImplementedError
    
    @compatibility(CompatibilityLevel.NONE)
    def obs_measured_turbulence_intensity_percent(self, t_s: np.float64) -> float:
        """ Observes the current turbulence intensity of the turbine.

        Args:
            t_s (np.float64): Current simulation time in seconds.

        Returns:
            float: Current turbulence intensity of the turbine (%).
        """
        raise NotImplementedError
    
    @compatibility(CompatibilityLevel.NONE)
    def obs_measured_nacelle_wind_speed_mps(self, t_s: np.float64) -> float:
        """ Observes the current nacelle-measured wind speed of the turbine.

        Args:
            t_s (np.float64): Current simulation time in seconds.

        Returns:
            float: Current nacelle-measured wind speed of the turbine (m/s).
        """
        raise NotImplementedError
    
    @compatibility(CompatibilityLevel.NONE)
    def obs_measured_nacelle_wind_dir_deg(self, t_s: np.float64) -> float:
        """ Observes the current nacelle-measured wind direction of the turbine.

        Args:
            t_s (np.float64): Current simulation time in seconds.

        Returns:
            float: Current nacelle-measured wind direction of the turbine (degrees).
        """
        raise NotImplementedError
    
    @compatibility(CompatibilityLevel.NONE)
    def obs_measured_yaw_orientation_deg(self, t_s: np.float64) -> float:
        """ Observes the current nacelle-measured yaw orientation of the turbine.

        Args:
            t_s (np.float64): Current simulation time in seconds.

        Returns:
            float: Current nacelle-measured yaw orientation of the turbine (degrees).
        """
        raise NotImplementedError
    
    """ 
    Location & Geometry 
    --------------------------------------- 
    """
    @abstractmethod
    @compatibility(CompatibilityLevel.NONE)
    def obs_turbine_base_location_m(self, t_s: np.float64 = 0.0) -> np.ndarray:
        """ Observes the location of the turbine.

        Args:
            t_s (np.float64): Current simulation time in seconds.
                                This measurement is only relevant for floating or airborne turbines, where the turbine base location may change over time due to platform motion.

        Returns:
            np.ndarray: Location of the turbine (x, y, z) in meters.
        """
        raise NotImplementedError

    @abstractmethod
    @compatibility(CompatibilityLevel.NONE)
    def obs_rotor_center_location_m(self, t_s: np.float64 = 0.0) -> np.ndarray:
        """ Observes the location of the rotor center of the turbine.
        This measurement includes the hub height of the turbine and the nacelle dimensions.

        Args:
            t_s (np.float64): Current simulation time in seconds.
                                Only relevant for floating or airborne turbines, where the rotor center location may change over time due to platform motion.

        Returns:
            np.ndarray: Location of the rotor center of the turbine (x, y, z) in meters.
        """
        raise NotImplementedError

    @abstractmethod
    @compatibility(CompatibilityLevel.NONE)
    def obs_rotor_hub_height_m(self, t_s: np.float64 = 0.0) -> float:
        """ Observes the hub height of the turbine.

        Args:
            t_s (np.float64): Current simulation time in seconds.
                                Only relevant for floating or airborne turbines, where the hub height may change over time due to platform motion.

        Returns:
            float: Hub height of the turbine (m).
        """
        raise NotImplementedError
    
    @compatibility(CompatibilityLevel.NONE)
    def obs_rotor_overhang_m(self) -> float:
        """ Observes the rotor overhang of the turbine.
        The rotor overhang is the distance from the rotor center to the tower centerline.

        Returns:
            float: Rotor overhang of the turbine (m).
        """
        raise NotImplementedError
    
    @abstractmethod
    @compatibility(CompatibilityLevel.NONE)
    def obs_rotor_diameter_m(self) -> float:
        """ Observes the rotor diameter of the turbine.

        Returns:
            float: Rotor diameter of the turbine (m).
        """
        raise NotImplementedError

    @compatibility(CompatibilityLevel.FULL)
    def obs_rotor_radius_m(self) -> float:
        """ Observes the rotor radius of the turbine.

        Returns:
            float: Rotor radius of the turbine (m).
        """
        return self.obs_rotor_diameter_m() / 2.0
    
    @compatibility(CompatibilityLevel.NONE)
    def obs_rotor_tilt_deg(self, t_s: np.float64 = 0.0) -> float:
        """ Observes the rotor tilt of the turbine.

        Args:
            t_s (np.float64): Current simulation time in seconds.

        Returns:
            float: Rotor tilt of the turbine (degrees).
        """
        raise NotImplementedError
    
    @abstractmethod
    @compatibility(CompatibilityLevel.NONE)
    def obs_rotor_yaw_orientation_deg(self, t_s: np.float64) -> float:
        """ Observes the current yaw orientation of the turbine.

        Args:
            t_s (np.float64): Current simulation time in seconds.

        Returns:
            float: Current yaw orientation of the turbine (degrees).
        """
        raise NotImplementedError
    
    @compatibility(CompatibilityLevel.FULL)
    def obs_num_blades(self) -> np.uint8:
        """ Observes the number of blades of the turbine.

        Returns:
            np.uint8: Number of blades of the turbine.
        """
        return np.uint8(3)  # Default implementation assumes a 3-bladed turbine. Override in derived classes if different.

    
    
    
