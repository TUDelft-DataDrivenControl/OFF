<p align="center">
  <img width="880" height="240" src="https://github.com/TUDelft-DataDrivenControl/OFF/blob/main/99_Design/01_Logo/OFF_Logo_wide.png">
</p>

# OFF
Welcome to the working repo for a dynamic parametric wake toolbox which combines and connects OnWaRDS, FLORIDyn and FLORIS (OFF).
## Purpose
The OFF toolbox is meant to provide one interface to dynamic parametric wake modelling. The goal is to enable testing of different approaches, comparisons using the same interface and a environment to develop new approaches.

## Development
### Current state
So what can you expect today?
The toolbox allows you to run dynamic wind farm simulations using a prescribed (uniform) flow field. The code uses an implementation of the FLORIDyn-framework \[1\], which then interfaces to the FLORIS model \[2\].

Current degrees of design-freedom include:
- The flow field can change uniformly in direction and speed, 
- the FLORIS wake can be changed, 
- the number and location of turbines can be altered.

Current limitations include:
- Heterogeneous flow field behaviour is not included, 
- Wake steerig and induction control is not included yet (soon to be added), 
- The dynamic solver is a preliminary implementation, and cannot be changed
  -  OnWaRDS approach not yet impemented
  -  Particle model and behaviour not yet changable

Features include:
- Logging is impemented
- There is an auto-documentation framework
- UML diagram of the code
- Measurements are stored in run-folders
- Effective wind speed flow field plots are possible (calculates the wind speed at a virtual turbine at a given grid point, facing the wind direction)

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
  - Create an OFF interface object to automate simulations
  - Communicate back what model has been used and under which conditions (Report)

### Longterm vision
FLORIDyn and OnWaRDS are hopefully only the start to create a transparent toolbox to explore dynamic parametric wake modelling choices. We therefore plan to implement more model-design decisions and allow a simple, automatable change of them.
Extensions of the code could include floating turbine descriptions, new wake solvers or modifications of existing ones.

**We do explicitly encourage the interested developers to engage and also to add their ideas and code to the toolbox.**

## How to run a simulation

##





## Contact, sources and citation
### Core Developers
Marcus Becker, Maxime Lejeune
### Sources
\[1\] 
\[2\]
\[3\]
### Citation
