
import numpy as np

# from geometric_yaw.geometric_yaw import calculate_geomYaw_ExpCorr

class WFFCstrategy():

    """
    A class to define a Wind Farm Flow Control (WFFC) strategy.

    This class initializes and manages strategy-specific parameters and specifies 
    the optimization methods. It is used within the class MSR_optimizer which 
    performs the optimization of different WFFC strategies.

    Attributes (general)
    -------------
        str_name : str
            Name of the control strategy
        var_name : str
            Name of the control variable associated with the strategy
        opt_method : str
            Optimization method, one of ['Refine', 'Discrete', 'Geometric yaw']

    Attributes (for opt_method 'Refine')
    -------------
        n_values : int
            Number of values tested at each iteration
        cmin : float
            Lower bound of the control variable
        cmax : float
            Upper bound of the control variable
    
    Attributes (for opt_method 'Discrete')
    -------------
        c_values_array : ndarray
            1D array of values to test at each iteration

    Attributes (for opt_method 'Geometric yaw')
    -------------
        geom_yaw_method : str
            Method for geometric yaw calculation
        ws_rated : float
            Rated wind speed
        diameter : float
            Turbine rotor diameter
        ws_eff : ndarray
            Effective wind speed per turbine
        yaw_max : float, optional
            Parameter for Exponential corrected method (default is 21.846, tuned for IEA22MW)
        p_x : float, optional
            Parameter for Exponential corrected method (default is 4.889, tuned for IEA22MW)
        p_y : float, optional
            Parameter for Exponential corrected method (default is 9.594, tuned for IEA22MW)
        q_x : float, optional
            Parameter for Exponential corrected method (default is 5.820, tuned for IEA22MW)
        q_y : float, optional
            Parameter for Exponential corrected method (default is 0.380, tuned for IEA22MW)
        alpha_f_ws_eff : float, optional
            Parameter for Exponential corrected method (default is 0.150, tuned for IEA22MW)
        w_corr : float, optional
            Parameter for Exponential corrected method (default is 0.456, tuned for IEA22MW)
    """


    def __init__(self,
                 str_name,
                 var_name,
                 opt_method,
                 **kwargs
                 ):
        
        """
        Parameters (general)
        -------------
            str_name : str
                Name of the control strategy
            var_name : str
                Name of the control variable associated with the strategy
            opt_method : str
                Optimization method, one of ['Refine', 'Discrete', 'Geometric yaw']

        Parameters (for opt_method 'Refine')
        -------------
            n_values : int
                Number of values tested at each iteration
            cmin : float
                Lower bound of the control variable
            cmax : float
                Upper bound of the control variable
        
        Parameters (for opt_method 'Discrete')
        -------------
            c_values_array : ndarray
                1D array of values to test at each iteration

        Parameters (for opt_method 'Geometric yaw')
        -------------
            geom_yaw_method : str
                Method for geometric yaw calculation
            ws_rated : float
                Rated wind speed
            diameter : float
                Turbine rotor diameter
            ws_eff : ndarray
                Effective wind speed per turbine
            yaw_max : float, optional
                Parameter for Exponential corrected method (default is 21.846, tuned for IEA22MW)
            p_x : float, optional
                Parameter for Exponential corrected method (default is 4.889, tuned for IEA22MW)
            p_y : float, optional
                Parameter for Exponential corrected method (default is 9.594, tuned for IEA22MW)
            q_x : float, optional
                Parameter for Exponential corrected method (default is 5.820, tuned for IEA22MW)
            q_y : float, optional
                Parameter for Exponential corrected method (default is 0.380, tuned for IEA22MW)
            alpha_f_ws_eff : float, optional
                Parameter for Exponential corrected method (default is 0.150, tuned for IEA22MW)
            w_corr : float, optional
                Parameter for Exponential corrected method (default is 0.456, tuned for IEA22MW)
        """

        
        self.str_name = str_name
        self.var_name = var_name
        self.opt_method = opt_method

        if self.opt_method == 'Refine':
            self.n_values = kwargs.get('n_values')
            self.cmin = kwargs.get('cmin')
            self.cmax = kwargs.get('cmax')
            self.offset_cvalues = np.zeros(self.n_values)

        elif self.opt_method=='Discrete':
            self.c_values_array = kwargs.get('c_values_array')

        elif self.opt_method=='Geometric yaw':
            self.geom_yaw_method = kwargs.get('geom_yaw_method')

            if self.geom_yaw_method == 'Exponential corrected':
                self.ws_rated = kwargs.get('ws_rated')
                self.diameter = kwargs.get('diameter')
                self.ws_eff = kwargs.get('ws_eff')                          # expected dim: (n_wt,)
                self.yaw_max = kwargs.get('yaw_max',21.846)                 # coefficient tuned for IEA22MW
                self.p_x = kwargs.get('p_x',4.889)                          # coefficient tuned for IEA22MW
                self.p_y = kwargs.get('p_y',9.594)                          # coefficient tuned for IEA22MW
                self.q_x = kwargs.get('q_x',5.820)                          # coefficient tuned for IEA22MW
                self.q_y = kwargs.get('q_y',0.380)                          # coefficient tuned for IEA22MW
                self.alpha_f_ws_eff = kwargs.get('alpha_f_ws_eff',0.150)    # coefficient tuned for IEA22MW
                self.w_corr = kwargs.get('w_corr',0.456)                    # coefficient tuned for IEA22MW

            else:
                raise TypeError("Geometric yaw method not available")

        else:
            raise TypeError("Optimization method not available")


    def calculate_geometric_yaw(self,x,y,wd):

        """This method calculates the geometric yaw

        Parameters
        -------------
            x : ndarray
                1D array containing x-coord of the turbines
            y : ndarray
                1D array containing y-coord of the turbines
            wd : float
                Wind direction

        Returns
        -------------
        ndarray
            1D array containing the yaw angles, dim: (i,)
        """

        if self.geom_yaw_method == 'Exponential corrected':
            yaw_ilk = calculate_geomYaw_ExpCorr(x,
                                                y,
                                                wd = np.array([wd]),
                                                ws = np.array([8.]),
                                                ws_rated = self.ws_rated,
                                                diameter = self.diameter,
                                                ws_eff = self.ws_eff,
                                                yaw_max = self.yaw_max,
                                                p_x = self.p_x,
                                                p_y = self.p_y,
                                                q_x = self.q_x,
                                                q_y = self.q_y,
                                                alpha_f_ws_eff = self.alpha_f_ws_eff,
                                                w_corr = self.w_corr
                                                )
            yaw_array_i = yaw_ilk.reshape(-2)
        else:
            raise TypeError('calculate_geometric_yaw can be called only when the optimation method is set to Geometric yaw')
        return yaw_array_i




class MSR_optimizer():

    """
    A class to define the Multi-strategy Serial Refine (MSR) optimizer.

    This class initializes the MSR optimizer, allows to add different control 
    strategies (defined with the class WWCstrategy), and to run the WFFC optimzation.

    Indices of the ndarrays
        s: step of iteration
        i: index for the turbine
        j: index for the startegy


    Attributes
    -------------
        x : ndarray
            1D array containing x-coord of the turbines
        y : ndarray
            1D array containing y-coord of the turbines
        wd : float
            Wind direction
        f_obj : ObjFuncComponent object
            Objective function
        n_step : int, optional
            Number of steps over which the farm is iterated (default is 3)
        exclusivity : bool, optional
            If True, the simultaneous operation of two strategies on the same turbine is not allowed (default is True)
        c_opt_global_ij : ndarray
            2D array containing the optimal control variables, dim=(i,j)
        c_opt : dict
            Dictionary containing the optimal control varibales and the corresponding control variable name
        f_opt : float
            Objective function optimal value
    """

    
    def __init__(self,
                 x,
                 y,
                 wd,
                 f_obj,
                 n_step = 3,
                 exclusivity = True
                 ):
            
        """
        Parameters
        -------------
        x : ndarray
            1D array containing x-coord of the turbines
        y : ndarray
            1D array containing y-coord of the turbines
        wd : float
            Wind direction
        f_obj : ObjFuncComponent object
            Objective function
        n_step : int, optional
            Number of steps over which the farm is iterated (default is 3)
        exclusivity : bool, optional
            If True, the simultaneous operation of two strategies on the same turbine is not allowed (default is True)
        """

        self.x = x
        self.y = y
        self.n_wt = len(x)
        self.wd = wd
        self.f_obj = f_obj

        self.str_list = []
        self.var_list = []
        self.n_strategy = len(self.str_list)

        self.n_step = n_step
        self.exclusivity = exclusivity
        
    
    def add_strategy(self,str_name,var_name,opt_method,**kwargs):

        """ This method adds a control strategy to the optimizer.
        
        Parameters
        -------------
        str_name : str
            Name of the control strategy
        var_name : str
            Name of the control variable associated with the strategy
        opt_method : str
            Optimization method, one of ['Refine', 'Discrete', 'Geometric yaw']
        Keyword Args:
            Depending on the optimization method
        """
        
        wffc_strategy = WFFCstrategy(str_name,var_name,opt_method,**kwargs)
        self.str_list = self.str_list+[wffc_strategy]
        self.var_list = self.var_list+[var_name]
        self.n_strategy = len(self.str_list)


    def _evaluate_f_obj(self,c_input_ij):

        """ This method adds a control strategy to the optimizer.
        
        Parameters
        -------------
        c_input_ij : ndarray
            2D array containing the values of the control variables, dim: (i,j)

        Returns
        -------------
        float
            Objective function value
        """

        c_input_i_list = list(c_input_ij.T)
        c_input_dict = dict(zip(self.var_list,c_input_i_list))
        f_val = self.f_obj(**c_input_dict)
        return f_val
        

    def optimize(self):

        """ This method runs the optimization.
        
        """

        # order the turbines depending on the wind direction
        theta = np.pi*(270-self.wd)/180
        x_rot = self.x*np.cos(theta)+self.y*np.sin(theta)
        ind_turbine_ordered = np.argsort(x_rot)
        
        # general initialization
        self.c_opt_global_ij =  np.zeros((self.n_wt,self.n_strategy),dtype=float)
        f_0 = self._evaluate_f_obj(self.c_opt_global_ij)
        f_opt_local_j = np.ones(self.n_strategy)*f_0
        c_opt_local_ijj =  np.zeros((self.n_wt,self.n_strategy,self.n_strategy),dtype=float)


        # initialization of each control strategy
        for j in np.arange(self.n_strategy):

            if self.str_list[j].opt_method == 'Refine':
                self.str_list[j].offset_cvalues = np.linspace(self.str_list[j].cmin,self.str_list[j].cmax,self.str_list[j].n_values,endpoint=True)

            elif self.str_list[j].opt_method=='Discrete':
                ... # no need for initialization

            elif self.str_list[j].opt_method=='Geometric yaw':
                yaw_array_i = self.str_list[j].calculate_geometric_yaw(self.x,self.y,self.wd)


        # iterate for each step
        for s in np.arange(self.n_step):
            
            # iterate for each turbine
            for i in np.arange(self.n_wt):
            
                # iterate for each strategy
                for j in np.arange(self.n_strategy):

                    # select control values to test
                    if self.str_list[j].opt_method == 'Refine':
                        c_values_test = c_opt_local_ijj[ind_turbine_ordered[i],j,j]+self.str_list[j].offset_cvalues
                        c_values_test = np.minimum(c_values_test,self.str_list[j].cmax)
                        c_values_test = np.maximum(c_values_test,self.str_list[j].cmin)

                    elif self.str_list[j].opt_method=='Discrete':
                        c_values_test = self.str_list[j].c_values_array

                    elif self.str_list[j].opt_method=='Geometric yaw':
                        c_values_test = np.array([yaw_array_i[ind_turbine_ordered[i]],0.])


                    # iterate for each value to test
                    for c in c_values_test:
                                
                        # test new control matrix
                        c_test_ij = self.c_opt_global_ij.copy()
                        if self.exclusivity:
                            c_test_ij[ind_turbine_ordered[i],:] = 0
                        c_test_ij[ind_turbine_ordered[i],j] = c
                        f_test = self._evaluate_f_obj(c_test_ij)
                        
                        # check improvement
                        if f_test>f_opt_local_j[j]:
                            f_opt_local_j[j] = f_test
                            c_opt_local_ijj[:,:,j] = c_test_ij.copy()
                            f_opt_local_j[j] = f_test

                # cooridnation block: find best control strategy
                j_opt = np.argmax(f_opt_local_j)
                self.f_opt = f_opt_local_j[j_opt]
                self.c_opt_global_ij[ind_turbine_ordered[i],:] = c_opt_local_ijj[ind_turbine_ordered[i],:,j_opt]



            # refinement block: update control values for the next step
            for j in np.arange(self.n_strategy):

                if self.str_list[j].opt_method == 'Refine':
                    cdelta = (np.max(self.str_list[j].offset_cvalues)-np.min(self.str_list[j].offset_cvalues))/(2*(self.str_list[j].n_values-1))
                    self.str_list[j].offset_cvalues = np.linspace(-cdelta,cdelta,self.str_list[j].n_values,endpoint=True)

                elif self.str_list[j].opt_method=='Discrete':
                    ... # no need of refinement

                elif self.str_list[j].opt_method=='Geometric yaw':
                    ... # no need of refinement


            # save values
            c_opt_i_list = list(self.c_opt_global_ij.T)
            self.c_opt = dict(zip(self.var_list,c_opt_i_list))




class ObjFuncComponent():

    """
    A class to call the objective function as function of the control variables .

    Attributes
    -------------
        obj_func : func
            Objective function
        input_keys : str
            Name of teh control variables
        Keyword Args
            Addioitnal args of the objective function
    """


    def __init__(self,obj_func,input_keys,**kwargs):
        
        """
        Parameters
        -------------
            obj_func : func
                Objective function
            input_keys : str
                Name of teh control variables
            Keyword Args
                Addioitnal args of the objective function
        """

        self.obj_func = obj_func
        self.input_keys = input_keys
        self.kwargs = kwargs


    def __call__(self,**c_input_dict):

        """ This method evakuates the objective function based on the control variables.

        Parameters
        -------------
            Keyword Args
                Control variables

        Returns
        -------------
        float
            Objective function value
        """

        return self.obj_func(**{**self.kwargs,**c_input_dict})



#%% TODO:
# - add additional TypeError (e.g. check if the nput keys are actually the control variables)
# - check if still works in case exclusisvity=False -> consider testing all the combination of values between the different stregy (more expensive)
# - create a Multi-objective version where the Pareto front is accumulated at each iteration
# - allows the optimizaiton of timeseries (refinement along time as well)







