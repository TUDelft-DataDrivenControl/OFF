import numpy as np

from __future__ import annotations

from abc import ABC, abstractmethod

class TurbineModel(ABC):
    """Base interface for turbine aerodynamics/dynamics."""

    @abstractmethod
    def step(self, dt: float, effective_wind_speed_mps: float) -> None:
        raise NotImplementedError

    @abstractmethod
    def set_yaw(self, yaw_deg: float) -> None:
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
    
    @abstractmethod
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

    @abstractmethod
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
    
    @abstractmethod
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
    
    @abstractmethod
    def obs_thrust_force_n(self, t_s: np.float64) -> float:
        """ Observes the current thrust force of the turbine.

        Args:
            t_s (np.float64): Current simulation time in seconds.

        Returns:
            float: Current thrust force of the turbine (N).
        """
        raise NotImplementedError
    

    """ 
    Torque
    --------------------------------------- 
    """
    @abstractmethod
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
    @abstractmethod
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

    @abstractmethod
    def obs_collective_pitch_angle_deg(self, t_s: np.float64) -> float:
        """ Observes the current collective pitch angle of the turbine.

        Args:
            t_s (np.float64): Current simulation time in seconds.

        Returns:
            float: Current collective pitch angle of the turbine (degrees).
        """
        raise NotImplementedError
    

    """ 
    Measurements & Sensors 
    --------------------------------------- 
    """
    @abstractmethod
    def obs_measured_rotor_averaged_wind_speed_mps(self, t_s: np.float64) -> float:
        """ Observes the current rotor-averaged wind speed of the turbine.

        Args:
            t_s (np.float64): Current simulation time in seconds.

        Returns:
            float: Current rotor-averaged wind speed of the turbine (m/s).
        """
        raise NotImplementedError
    
    @abstractmethod
    def obs_measured_rotor_averaged_wind_dir_deg(self, t_s: np.float64) -> float:
        """ Observes the current rotor-averaged wind direction of the turbine.

        Args:
            t_s (np.float64): Current simulation time in seconds.

        Returns:
            float: Current rotor-averaged wind direction of the turbine (degrees).
        """
        raise NotImplementedError
    
    @abstractmethod
    def obs_measured_turbulence_intensity_percent(self, t_s: np.float64) -> float:
        """ Observes the current turbulence intensity of the turbine.

        Args:
            t_s (np.float64): Current simulation time in seconds.

        Returns:
            float: Current turbulence intensity of the turbine (%).
        """
        raise NotImplementedError
    
    @abstractmethod
    def obs_measured_nacelle_wind_speed_mps(self, t_s: np.float64) -> float:
        """ Observes the current nacelle-measured wind speed of the turbine.

        Args:
            t_s (np.float64): Current simulation time in seconds.

        Returns:
            float: Current nacelle-measured wind speed of the turbine (m/s).
        """
        raise NotImplementedError
    
    @abstractmethod
    def obs_measured_nacelle_wind_dir_deg(self, t_s: np.float64) -> float:
        """ Observes the current nacelle-measured wind direction of the turbine.

        Args:
            t_s (np.float64): Current simulation time in seconds.

        Returns:
            float: Current nacelle-measured wind direction of the turbine (degrees).
        """
        raise NotImplementedError
    
    @abstractmethod
    def obs_measured_rotor_yaw_orientation_deg(self, t_s: np.float64) -> float:
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
    
    @abstractmethod
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

    @abstractmethod
    def obs_rotor_radius_m(self) -> float:
        """ Observes the rotor radius of the turbine.

        Returns:
            float: Rotor radius of the turbine (m).
        """
        return self.obs_rotor_diameter_m() / 2.0
    
    @abstractmethod
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
    
    @abstractmethod
    def obs_rotor_yaw_misalignment_deg(self, t_s: np.float64) -> float:
        """ Observes the current yaw misalignment of the turbine.

        Args:
            t_s (np.float64): Current simulation time in seconds.

        Returns:
            float: Current yaw misalignment of the turbine (degrees).
        """
        raise NotImplementedError


    
    
    