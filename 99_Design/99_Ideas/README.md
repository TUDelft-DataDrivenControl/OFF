# Ideas for OFF
Preliminary collection of ideas relevant to OFF, sorted by topic

## Input
### Interactive Simulation Assembly
The interactive simulation assembly could be moved to a html based dashboard, such as 
[Plotly | Dash](https://dash.gallery/Portal/) (parially commercial), tutorial to use 
[Dash](https://realpython.com/python-dash/). Otherwise there is also [Binder](https://mybinder.org) 
(partially commercial) which converts a Git into a website. Maybe also [Django](https://www.djangoproject.com/start/overview/). 

### Database
The IEA Wind Task 37 has defined some standards for .yaml files in their [GitHub](https://github.com/IEAWindTask37/windIO).
These standards aim to allow for interchangeable components in an optimization framework. They also allow to check new 
data for incompleteness.

The OFF requirements are a bit different from what the Task has defined. It would make sense to find these differences
and to set up a similar, hopefully compatible, framework for the database:
- OFF requires (profits from) additional DOF for the turbine data - for instance CT based on blade pitch and tip-speed-ratio
instead of wind speed, assuming that the turbine does not run greedy control all the time.
- Wind field input has to facilitate dynamic choices
- ...?

### Network based input
Currently the framework is set up to simulate based on prescribed inputs. In the future this could be replaced by a 
network link (UDP / TCP) to receive data from a real wind farm or a *real wind farm*.

## Simulation

### Wakes
dynamic surrogate wakes
### Wake Solvers
Julia & C based wake solvers

## Logging
### Report
Generate a LaTeX based report based on the logging files and results.

## Testing
Actually implement it
## Visualization
### Show case simulations
Live inputs which show impacts in real time. Changing the yaw angle for instance or changing the wind direction etc.
This would be great if linked with a great visualization engine such as *Unreal Engine 5*.

## Use cases
### Pre-Simulation for LES
Enhance AMR Wind GUI with OFF pre-simulation

## Documentation


