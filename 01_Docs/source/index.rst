

.. image:: ../../99_Design/01_Logo/OFF_Logo_wide.png
   :width: 100%

a dynamic parametric wake toolbox which combines and connects OnWaRDS [1]_, FLORIDyn [2]_ and FLORIS [3]_

Documentation
=============

Run the main.py in the `03_Code <https://github.com/TUDelft-DataDrivenControl/OFF/tree/main/03_Code>`_ folder.

To change the simulation, you have to change the .yaml file OFF calls. This is defined by one of the first limes of code in the main function. The .yaml structure is showed in `run_example.yaml <https://github.com/TUDelft-DataDrivenControl/OFF/blob/main/02_Examples_and_Cases/02_Example_Cases/run_example.yaml>`_ . This is where you can change the wind farm layout, the flow conditions, the wake model etc.

.. image:: media/9T_ueff_2x2.png
   :width: 75%

.. image:: media/Results_power_9T.png
   :width: 75%

Code
----

The documentation is handled automatically using Sphinx and available here.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

.. toctree::
   :maxdepth: 1

   off.off
   off.turbine
   off.windfarm
   off.states
   off.ambient
   off.observation_points
   off.utils

License
-------
The documentation for this program is under a creative commons attribution share-alike 4.0 license. http://creativecommons.org/licenses/by-sa/4.0/

Publications
------------

.. [1] FLORIDyn - A dynamic and flexible framework for real-time wind farm control, M. Becker, D. Allaerts, J.W. van Wingerden, 2022 J. Phys.: Conf. Ser. 2265(2022) 032103

.. [2] FLORIS Wake Modeling and Wind Farm Controls Software, National Renewable Energy Laboratory, 2023, GitHub

.. [3] A Meandering-Capturing Wake Model Coupled to Rotor-Based Flow-Sensing for Operational Wind Farm Flow Prediction, M. Lejeune, M. Moens, P. Chatelain, 2022 Front. Energy Res., Sec. Wind Energy
