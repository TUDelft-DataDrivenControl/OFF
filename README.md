<p align="center">
  <img width="880" height="240" src="https://github.com/TUDelft-DataDrivenControl/OFF/blob/main/99_Design/01_Logo/OFF_Logo_wide_maintenance.png">
</p>

# OFF
Welcome to the working repo for a dynamic parametric wake toolbox which combines and connects OnWaRDS, FLORIDyn and FLORIS (OFF).
## Purpose
The OFF toolbox is meant to provide one interface to dynamic parametric wake modelling. The goal is to enable testing of different approaches, comparisons using the same interface and an environment to develop new approaches.

## Development
> [!INFO]
> OFF is under maintenance! We are working under the hood on a better interface and internal structure of the model.
The goal is to simplify the installation of the model, improve performance, make it easier to extend, add more examples, and much more.

> [!INFO]
> However, for these changes to happen, we may need to make some braking changes. This may mean that the code you wrote may not work with future versions anymore, and that additions you made may not fit anymore. If you have code additions, create a pull request and we'll try to merge them asap. If you have wishes for the model, create an issue.

### Current state
So what can you expect today?
The toolbox allows you to run dynamic wind farm simulations using a prescribed (uniform) flow field. The code uses an implementation of the FLORIDyn-framework \[1\], which then interfaces to the FLORISv4 model \[2\].

*Current degrees of design-freedom include:*
- The flow field can change uniformly in direction and speed, 
- the FLORISv4 wake can be changed, 
- the number and location of turbines can be altered,
- Dead-band LuT Wake steerig control is available on a controller test branch

*Current limitations include:*
- Heterogeneous flow field behaviour is not included, 
- There is only one dynamic solver which cannot be changed
  -  OnWaRDS \[3\] approach not yet impemented
  -  Particle model and behaviour not yet changable

*Features include:*
- Logging is impemented
- There is an [auto-documentation framework](https://tudelft-datadrivencontrol.github.io/OFF/)
- UML diagram of the code
- Measurements are stored in run-folders
- Preliminary effective wind speed flow field plots are possible

### Current development focus
1. Completion of the toolbox
  - Implement OnWaRDS
  - Implement a low-level controller for yaw and induction
  - Validation and reference cases
2. Features
  - Input flow field data (heterogeneous flow)
  - Improved flow field plotting
  - Computational performance
3. User friendliness
  - Complete the "Interactive Simulation Assembly"
  - Communicate back what model has been used and under which conditions (Report)

### Longterm vision
FLORIDyn and OnWaRDS are hopefully only the start to create a transparent toolbox to explore dynamic parametric wake modelling choices. We therefore plan to implement more model-design decisions and allow a simple, automatable change of them.
Extensions of the code could include floating turbine descriptions, new wake solvers or modifications of existing ones.

## Contributing
Fork branch commit push pull!
**We do explicitly encourage the interested developers to engage and also to add their ideas and code to the toolbox.**
The Git is also meant as a place to discuss and propose new changes.

## How to run a simulation
Run the main.py in the 03_Code folder.

To change the simulation, you have to change the .yaml file OFF calls. This is defined by one of the first limes of code in the main function. The .yaml structure is showed in 02_Examples_and_Cases/02_Example_Cases/run_example.yaml . This is where you can change the wind farm layout, the flow conditions, the wake model etc.

A thorough description of the code is available [here](https://tudelft-datadrivencontrol.github.io/OFF/). The documentation is handled automatically using Sphinx.

## Contact and sources
### Core Developers
[Marcus Becker](https://www.tudelft.nl/staff/marcus.becker/?cHash=4e16fc5842bde9873a2a322dcbc17453) (TU Delft) and 
[Maxime Lejeune](https://uclouvain.be/fr/repertoires/maxime.lejeune) (UCLouvain) are the current core developers of the OFF toolbox.
### Sources
\[1\] FLORIDyn - A dynamic and flexible framework for real-time wind farm control, M. Becker, D. Allaerts, J.W. van Wingerden, 2022 J. Phys.: Conf. Ser. 2265(2022) 032103

\[2\] FLORIS Wake Modeling and Wind Farm Controls Software, National Renewable Energy Laboratory, 2024, GitHub

\[3\] A Meandering-Capturing Wake Model Coupled to Rotor-Based Flow-Sensing for Operational Wind Farm Flow Prediction, M. Lejeune, M. Moens, P. Chatelain, 2022 Front. Energy Res., Sec. Wind Energy
