import numpy as np

from __future__ import annotations

from abc import ABC, abstractmethod

from Utils import *
class TurbineModel(ABC):
    """Base interface for turbine aerodynamics/dynamics."""

    @abstractmethod
    def step(self, it: int) -> None:
        """ Advances the Turbine model by a given number of iterations.

        Args:
            it (int): Current iteration of the simulation. The current real time since simulation start is it * dt, where dt is the global time step.
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError
    
    def get_citation(self) -> str:
        """ Returns a citation string for the turbine model. Default implementation returns a generic citation.

        Returns:
            str: Citation string for the turbine model.
        """
        return (
            "@software{floridyn_off_2026,\n"
            "  author       = {Becker, Marcus and Lejeune, Maxime and van Straalen, Ivo},\n"
            "  title        = {OFF wind farm simulation toolbox},\n"
            "  year         = {2026},\n"
            "  version      = {1.0.0},\n"
            "  publisher    = {GitHub},\n"
            "  url          = {https://github.com/TUDelft-DataDrivenControl/OFF}\n"
            "}\n\n"
            "@article{becker2025ADynamicModel,\n"
            "  author       = {Becker, Marcus and Lejeune, Maxime and Chatelain, Philippe and Allaerts, Dries and Mudafort, Rafael and van Wingerden, Jan-Willem},\n"
            "  title        = {A dynamic open-source model to investigate wake dynamics in response to wind farm flow control strategies},\n"
            "  journal      = {Wind Energy Science},\n"
            "  year         = {2025},\n"
            "  volume       = {10},\n"
            "  pages        = {1055--1075},\n"
            "  doi          = {10.5194/wes-10-1055-2025}\n"
            "}"
        )
    
    def req_describe(self) -> dict[str, SupportType]:
        """ Returns a dictionary describing the atmospheric model. Default implementation returns an empty dictionary.

        Returns:
            dict[str, Any]: Dictionary describing the atmospheric model.
        """
        return {
            "obs_generator_power_w": SupportType.NOT_SUPPORTED,
            "obs_aerodynamic_power_w": SupportType.NOT_SUPPORTED,
            "obs_available_power_w": SupportType.NOT_SUPPORTED,
            "obs_power_coefficient": SupportType.NOT_SUPPORTED,
            "obs_power_curve": SupportType.NOT_SUPPORTED,
            "obs_thrust_coefficient": SupportType.NOT_SUPPORTED,
            "obs_thrust_force_n": SupportType.NOT_SUPPORTED,
            "obs_thrust_curve": SupportType.NOT_SUPPORTED,
            "obs_aerodynamic_torque_nm": SupportType.NOT_SUPPORTED,
            "obs_generator_torque_nm": SupportType.NOT_SUPPORTED,
            "obs_rotor_speed_radps": SupportType.NOT_SUPPORTED,
            "obs_rotor_speed_rpm": SupportType.NOT_SUPPORTED,
            "obs_collective_pitch_angle_deg": SupportType.NOT_SUPPORTED,
            "obs_individual_pitch_angles_deg": SupportType.NOT_SUPPORTED,
            "obs_measured_rotor_averaged_wind_speed_mps": SupportType.NOT_SUPPORTED,
            "obs_measured_rotor_averaged_wind_dir_deg": SupportType.NOT_SUPPORTED,
            "obs_measured_turbulence_intensity_percent": SupportType.NOT_SUPPORTED,
            "obs_measured_nacelle_wind_speed_mps": SupportType.NOT_SUPPORTED,
            "obs_measured_nacelle_wind_dir_deg": SupportType.NOT_SUPPORTED,
            "obs_measured_yaw_orientation_deg": SupportType.NOT_SUPPORTED,
            "obs_turbine_base_location_m": SupportType.NOT_SUPPORTED,
            "obs_rotor_center_location_m": SupportType.NOT_SUPPORTED,
            "obs_rotor_hub_height_m": SupportType.NOT_SUPPORTED,
            "obs_rotor_overhang_m": SupportType.NOT_SUPPORTED,
            "obs_rotor_diameter_m": SupportType.NOT_SUPPORTED,
            "obs_rotor_radius_m": SupportType.NOT_SUPPORTED,
            "obs_rotor_tilt_deg": SupportType.NOT_SUPPORTED,
            "obs_rotor_yaw_orientation_deg": SupportType.NOT_SUPPORTED,
            "obs_num_blades": SupportType.NOT_SUPPORTED
        }
    
    def req_check_component(self, component) -> bool:
        """ Checks if the given component is compatible with the turbine controller. Default implementation returns False.

        Args:
            component: Component to check for compatibility.

        Returns:
            bool: True if the component is compatible, False otherwise.
        """
        return False

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
    def obs_thrust_coefficient(self, t_s: np.float64) -> float:
        """ Observes the current thrust coefficient of the turbine.

        Args:
            t_s (np.float64): Current simulation time in seconds.

        Returns:
            float: Current thrust coefficient of the turbine (dimensionless).
        """
        raise NotImplementedError
    
    def obs_thrust_force_n(self, t_s: np.float64) -> float:
        """ Observes the current thrust force of the turbine.

        Args:
            t_s (np.float64): Current simulation time in seconds.

        Returns:
            float: Current thrust force of the turbine (N).
        """
        raise NotImplementedError
    
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
    def obs_aerodynamic_torque_nm(self, t_s: np.float64) -> float:
        """ Observes the current aerodynamic torque of the turbine.

        Args:
            t_s (np.float64): Current simulation time in seconds.

        Returns:
            float: Current aerodynamic torque of the turbine (Nm).
        """
        raise NotImplementedError
    
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
    def obs_rotor_speed_radps(self, t_s: np.float64) -> float:
        """ Observes the current rotor speed of the turbine.

        Args:
            t_s (np.float64): Current simulation time in seconds.

        Returns:
            float: Current rotor speed of the turbine (rad/s).
        """
        raise NotImplementedError
    
    def obs_rotor_speed_rpm(self, t_s: np.float64) -> float:
        """ Observes the current rotor speed of the turbine.

        Args:
            t_s (np.float64): Current simulation time in seconds.

        Returns:
            float: Current rotor speed of the turbine (RPM).
        """
        return self.obs_rotor_speed_radps(t_s) * 60.0 / (2.0 * 3.141592653589793)

    def obs_collective_pitch_angle_deg(self, t_s: np.float64) -> float:
        """ Observes the current collective pitch angle of the turbine.

        Args:
            t_s (np.float64): Current simulation time in seconds.

        Returns:
            float: Current collective pitch angle of the turbine (degrees).
        """
        raise NotImplementedError
    
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
    def obs_measured_rotor_averaged_wind_speed_mps(self, t_s: np.float64) -> float:
        """ Observes the current rotor-averaged wind speed of the turbine.

        Args:
            t_s (np.float64): Current simulation time in seconds.

        Returns:
            float: Current rotor-averaged wind speed of the turbine (m/s).
        """
        raise NotImplementedError
    
    def obs_measured_rotor_averaged_wind_dir_deg(self, t_s: np.float64) -> float:
        """ Observes the current rotor-averaged wind direction of the turbine.

        Args:
            t_s (np.float64): Current simulation time in seconds.

        Returns:
            float: Current rotor-averaged wind direction of the turbine (degrees).
        """
        raise NotImplementedError
    
    def obs_measured_turbulence_intensity_percent(self, t_s: np.float64) -> float:
        """ Observes the current turbulence intensity of the turbine.

        Args:
            t_s (np.float64): Current simulation time in seconds.

        Returns:
            float: Current turbulence intensity of the turbine (%).
        """
        raise NotImplementedError
    
    def obs_measured_nacelle_wind_speed_mps(self, t_s: np.float64) -> float:
        """ Observes the current nacelle-measured wind speed of the turbine.

        Args:
            t_s (np.float64): Current simulation time in seconds.

        Returns:
            float: Current nacelle-measured wind speed of the turbine (m/s).
        """
        raise NotImplementedError
    
    def obs_measured_nacelle_wind_dir_deg(self, t_s: np.float64) -> float:
        """ Observes the current nacelle-measured wind direction of the turbine.

        Args:
            t_s (np.float64): Current simulation time in seconds.

        Returns:
            float: Current nacelle-measured wind direction of the turbine (degrees).
        """
        raise NotImplementedError
    
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
    def obs_rotor_hub_height_m(self, t_s: np.float64 = 0.0) -> float:
        """ Observes the hub height of the turbine.

        Args:
            t_s (np.float64): Current simulation time in seconds.
                                Only relevant for floating or airborne turbines, where the hub height may change over time due to platform motion.

        Returns:
            float: Hub height of the turbine (m).
        """
        raise NotImplementedError
    
    def obs_rotor_overhang_m(self) -> float:
        """ Observes the rotor overhang of the turbine.
        The rotor overhang is the distance from the rotor center to the tower centerline.

        Returns:
            float: Rotor overhang of the turbine (m).
        """
        raise NotImplementedError
    
    @abstractmethod
    def obs_rotor_diameter_m(self) -> float:
        """ Observes the rotor diameter of the turbine.

        Returns:
            float: Rotor diameter of the turbine (m).
        """
        raise NotImplementedError

    def obs_rotor_radius_m(self) -> float:
        """ Observes the rotor radius of the turbine.

        Returns:
            float: Rotor radius of the turbine (m).
        """
        return self.obs_rotor_diameter_m() / 2.0
    
    def obs_rotor_tilt_deg(self, t_s: np.float64 = 0.0) -> float:
        """ Observes the rotor tilt of the turbine.

        Args:
            t_s (np.float64): Current simulation time in seconds.

        Returns:
            float: Rotor tilt of the turbine (degrees).
        """
        raise NotImplementedError
    
    @abstractmethod
    def obs_rotor_yaw_orientation_deg(self, t_s: np.float64) -> float:
        """ Observes the current yaw orientation of the turbine.

        Args:
            t_s (np.float64): Current simulation time in seconds.

        Returns:
            float: Current yaw orientation of the turbine (degrees).
        """
        raise NotImplementedError
    
    def obs_num_blades(self) -> np.uint8:
        """ Observes the number of blades of the turbine.

        Returns:
            np.uint8: Number of blades of the turbine.
        """
        return np.uint8(3)  # Default implementation assumes a 3-bladed turbine. Override in derived classes if different.

    
    
    