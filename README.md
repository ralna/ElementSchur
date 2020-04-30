# Element Schur complement preconditioners

This repository contains the code which produces the results for
An element-based preconditioner for mixed finite element problems, Tyrone Rees and Michael Wathen, 2020


## Install instructions

The only external dependence is Firedrake, which can be installed following the instructions from https://www.firedrakeproject.org/download.html. This will install the default settings. Once Firedrake is installed correctly then this will create a new python environment that needs to be activated.

To finish the installation, run
`pip install .`
from the root directory. This will install the repository plus the additional `pandas` requirement.

## Running

There are three examples presented here:
- Stokes flow
- Mixed formulation of Maxwell's equations
- Navier-Stokes equations
These examples are store in individual folders within the `ElementSchur/examples`. To find install instructions for each example run `python filename.py –help`.

Note: For the backwards facing step examples (for Stokes and Navier-Stokes) one will need to make the mesh files which are found in `ElementSchur/ElementSchur/mesh`. There are `Makefile` within these folders which will create the appropriate meshes. Firedrake uses the meshing software Gmsh https://gmsh.info/.

The script `ElementSchur/examples/run_script` was used to create the results presented in the paper.
