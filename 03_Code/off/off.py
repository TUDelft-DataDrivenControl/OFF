import os

import logging
lg = logging.getLogger('off')

import off.turbine as tur
import off.windfarm as wfm
import off.observation_points as ops
import off.ambient as amb
import numpy as np
import pandas as pd
import off.wake_solver as ws
from off.logger import CONSOLE_LVL, FILE_LVL, Formatter, _logger_add

from off import __file__ as OFF_PATH
OFF_PATH = OFF_PATH.rsplit('/',3)[0]

class OFF:
    """
    OFF is the central object which initializes the wind farm and runs the simulation
    """
    settings_sim = dict()
    wind_farm = wfm.WindFarm

    def __init__(self, wind_farm: wfm.WindFarm, settings_sim: dict, settings_wke: dict, settings_sol: dict, settings_cor: dict):
        self.wind_farm = wind_farm
        self.settings_sim = settings_sim
        self.__dir_init__( settings_sim )
        self.__logger_init__( settings_sim )
        settings_wke['sim_dir'] = self.root_dir
        # self.wake_solver = ws.FLORIDynTWFWakeSolver(settings_wke, settings_sol)
        self.wake_solver = ws.FLORIDynFlorisWakeSolver(settings_wke, settings_sol)

        if settings_cor['ambient']: 
            states_name = self.wind_farm.turbines[0].ambient_states.get_state_names()
            self.ambient_corrector =  amb.AmbientCorrector(settings_cor['ambient'], self.wind_farm.nT, states_name)

    def __get_runid__(self) -> int:        
        """ Extract and increment the run id

        Returns
        -------
        int
            Current run id.
        """
        run_id_path = f'{OFF_PATH}/03_Code/off/.runid'
        lg.info('RunID path: ' + run_id_path)

        try:
            fid = open(run_id_path)
        except FileNotFoundError:
            run_id = 0
        else: 
            with fid:
                run_id = int(fid.readline())

        with open(run_id_path, 'w') as fid:
            fid.write('{}'.format(run_id+1))

        lg.info(f'RunID: {run_id}')
        return run_id

    def __dir_init__(self, settings_sim: dict):
        """ Initialize the simulation folder and set the data path.

        Parameters
        ----------
        settings_sim : dict
            Dictionary containing the OFF parameters.

            Available options:
            
            :simulation folder:  *(str)*       - 
                Path to the folder where the simulation results and logs will be
                exported.Export directory name where figures and data are saved. 
                If ``''``, all exports are disabled; if None (by default), default 
                export directory name ``off_run_id``.
            :data folder:        *(str)*       - 
                Path to the folder containing the off data.
        """

        sim_dir  = settings_sim.setdefault('simulation folder', None)
        data_dir = settings_sim.get('data folder', None)

        run_id = self.__get_runid__()

        try:
            root_dir = data_dir or f'{os.environ["OFF_PATH"]}/runs/'
            lg.info('Root runs directory: ' + root_dir)
        except KeyError:
            root_dir = data_dir or f'{os.environ["PWD"][:-len("03_Code")]}/runs/'
            lg.warning('Initial root runs directory path retrieval was unsuccessful, used ' + root_dir)

        self.sim_dir = f'{root_dir}/off_run_{run_id}' if sim_dir is None else sim_dir
        self.root_dir = root_dir[:-len("runs/")]

        if not os.path.exists(self.sim_dir):
            os.makedirs(self.sim_dir)
            lg.info('Created simulation directory at ' + self.sim_dir)

    def __logger_init__(self, settings_sim: dict):
        """ Initializes the logger

        Parameters
        ----------
        settings_sim : dict
            Dictionary containing the OFF parameters
            
            :log console lvl:  *(str)*                  - 
                Logging level (``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``) used 
                for the console logging.
            :log file lvl:     *(str)*                  - 
                Logging level (``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``) used 
                for the file logging. Log logged to ``self.sim_dir/off.log``. 
                Not available if ``self.sim_dir`` not set.
        """
        file_lvl = settings_sim.setdefault('log file lvl', FILE_LVL).upper()
        console_lvl = settings_sim.setdefault('log console lvl', CONSOLE_LVL).upper()

        min_lvl = min(getattr(logging, file_lvl), getattr(logging, console_lvl))
        lg.setLevel(min_lvl)
        lg.propagate = False

        file_formatter = Formatter('%(levelname)s : %(filename)s, line %(lineno)d in %(funcName)s : %(message)s')
        console_formatter = Formatter('%(levelname)s : %(message)s')

        _logger_add(lg, logging.StreamHandler(), console_lvl, console_formatter)
        if self.sim_dir:
            if not self.sim_dir:
                lg.warning('Not simulation folder was specified: file logger disabled.')
            else: 
                log_fid = f'{self.sim_dir}/off.log'
                _logger_add(lg, logging.FileHandler(log_fid), file_lvl, file_formatter)
        
        lg.info(f'Saving data to {self.sim_dir}.')

    def init_sim(self, start_ambient: np.ndarray, start_turbine: np.ndarray):
        """
        Function which initializes the states within the ``self.wind_farm`` object. 
        Assigns turbine & ambient states and distributes the OPs downstream. OP 
        locations are not necessarily correct but the wakes are "unrolled" and 
        do not have to first develop.

        Parameters
        ----------
        start_ambient : np.ndarray
            1 x n vector with initial ambient state
        start_turbine : np.ndarray
            1 x n vector with initial turbine state
        """        

        for t in self.wind_farm.turbines:
            t.ambient_states.init_all_states(start_ambient)
            t.turbine_states.init_all_states(start_turbine)
            t.observation_points.init_all_states(t.ambient_states.get_turbine_wind_speed_u(),
                                                 t.ambient_states.get_turbine_wind_speed_v(),
                                                 t.get_rotor_pos(), self.settings_sim['time step'])

    def run_sim(self) -> pd.DataFrame:
        """
        Central function which executes the simulation and manipulates the ``self.wind_farm object``

        Returns
        -------
        pandas.Dataframe :
            Measurements from the entire simulation
        """
        lg.info(f'Running simulation from {self.settings_sim["time start"]} s to {self.settings_sim["time end"]} s.')
        lg.info(f'Time step: {self.settings_sim["time step"]} s.')

        # Allocate data structures for measurement (output), effective rotor wind speed (u,v) as well as OP speed
        m = pd.DataFrame()
        uv_r = np.zeros((len(self.wind_farm.turbines), 2))
        for t in np.arange(self.settings_sim['time start'],
                           self.settings_sim['time end'],
                           self.settings_sim['time step']):
            lg.info(f'Starting time step: {t} s.')

            # Predict - Get wind speeds at the rotor plane and to propagate the OPs
            for idx, tur in enumerate(self.wind_farm.turbines):
                uv_r[idx, :], uv_op, m_tmp = self.wake_solver.get_measurements(idx, self.wind_farm)
                m_tmp.t_idx = idx
                m_tmp['time'] = t
                m = pd.concat([m, m_tmp], ignore_index=True)
                tur.observation_points.set_op_propagation_speed(uv_op)

            lg.info(f'Rotor wind speed of all turbines:')
            lg.info(uv_r)

            # Correct
            self.ambient_corrector.update(t)
            for idx, tur in enumerate(self.wind_farm.turbines):
                self.ambient_corrector(idx, tur.ambient_states)

            # Control

            # Visualize

            # Predict - iterate all states
            for idx, tur in enumerate(self.wind_farm.turbines):
                tur.ambient_states.iterate_states_and_keep()
                tur.turbine_states.iterate_states_and_keep()
                tur.observation_points.propagate_ops(self.settings_sim['time step'])
                lg.debug(tur.observation_points.get_world_coord())

            lg.info(f'Ending time step: {t} s.')

        lg.info('Simulation finished. Resulting measurements:')
        lg.info(m)
        return m

    def set_wind_farm(self, new_wf: wfm.WindFarm):
        """
        Overwrite wind farm object with a new wind farm object. Can be used to restart the simulation from a given state

        Parameters
        ----------
        new_wf :  windfarm.WindFarm
            Wind farm object with turbines and states
        -------
        """
        self.wind_farm = new_wf

    def get_wind_farm(self) -> wfm.WindFarm:
        """
        Get the current wind farm object which equals the simulation state

        Returns
        -------
        windfarm.WindFarm :
            Wind farm object with turbines and states
        """
        return self.wind_farm

