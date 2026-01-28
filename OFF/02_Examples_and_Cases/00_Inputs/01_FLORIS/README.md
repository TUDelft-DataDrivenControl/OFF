# FLORIS files
The files in this folder have been retrieved from the FLORIS GitHub.
Full credit goes to NREL, FLORIS v4, 2022, https://github.com/NREL/floris

OFF assembles its own input file from the given "building block" yaml files. This means that e.g. 02_Wake only stores the wake part of the yaml input, other parts, such as logging and solver settings, are stored in the respective folders. Settings also relevant to OFF, such as the wind farm layout and the flow conditions are generated from the OFF input file.
Update to FLORIS v4, April 2024