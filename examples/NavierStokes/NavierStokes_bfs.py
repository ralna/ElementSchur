import argparse
import inspect
from firedrake import *
import os
import pprint

from ElementSchur import solver, navier_stokes, solver_options, utils

parser = argparse.ArgumentParser(add_help=True)
parser.add_argument('-s', '--schur', nargs='+',
                    default=['dual', 'pcd'],
                    help='Schur complement approximation type (default '
                    'dual reisz')
parser.add_argument('-N', '--N', type=int, required=True,
                    help='Number of mesh levels')
parser.add_argument('-Re', '--Re', type=float, default=1,
                    help='Viscosity of the fluid (default 1)')
parser.add_argument('-d', '--space-dim', type=str, default="2D",
                    help='Spacial dimension of the problem (default 2D)')
parser.add_argument('--plot-sol', type=str, default=False,
                    help='Plot solution (default False)')
parser.add_argument('--solve-type', type=str, default=False,
                    help='Defines the solver type to be direct or iterative '
                         '(default iterative)')
args, _ = parser.parse_known_args()


class BFS_problem_2D(navier_stokes.NavierStokes):

    def __init__(self, n, Re=1):
        super(BFS_problem_2D, self).__init__(n, Re)
        self.name = "ns_bfs_2d"
        base_path = os.path.dirname(inspect.getfile(navier_stokes))
        self.mesh_path = os.path.join(base_path, "mesh", "bfs_2d")

    def mesh_domain(self):
        file_name = os.path.join(self.mesh_path, f"bfs_2d_{self.n}.msh")
        print(file_name)
        if os.path.isfile(file_name):
            mesh = Mesh(file_name)
        else:
            raise RuntimeError("Mesh file does not exsist, please run make "
                               f"in {file_name}")
        return mesh

    def nullspace(self, Z):
        return None

    def poiseuille_flow(self):
        (x, y) = SpatialCoordinate(self.Z.mesh())
        return as_vector([4 * (2 - y) * (y - 1) * (y > 1), 0])

    def bcs(self):
        bcs = [DirichletBC(self.Z.sub(0), self.poiseuille_flow(), 1),
               DirichletBC(self.Z.sub(0), Constant((0., 0.)), 2)]
        return bcs

    def rhs(self):
        return Constant((0., 0.))


class BFS_problem_3D(navier_stokes.NavierStokes):

    def __init__(self, n, Re=1):
        super(BFS_problem_3D, self).__init__(n, Re)
        self.name = "ns_bfs_3d"

        base_path = os.path.dirname(inspect.getfile(navier_stokes))
        self.mesh_path = os.path.join(base_path, "mesh", "bfs_3d")

    def mesh_domain(self):
        file_name = os.path.join(self.mesh_path, f"bfs_3d_{self.n}.msh")
        if os.path.isfile(file_name):
            mesh = Mesh(file_name)
        else:
            raise RuntimeError("Mesh file does not exist, please run make "
                               f"in {file_name}")
        return mesh

    def nullspace(self, Z):
        return None

    def poiseuille_flow(self):
        (x, y, z) = SpatialCoordinate(self.Z.mesh())
        out = as_vector([16 * (2 - y) * (y - 1) * z * (1 - z) * (y > 1), 0, 0])
        return out

    def bcs(self):
        bcs = [DirichletBC(self.Z.sub(0), self.poiseuille_flow(), 1),
               DirichletBC(self.Z.sub(0), Constant((0., 0., 0.)), 3)]
        return bcs

    def rhs(self):
        return Constant((0., 0., 0.))


schur = args.schur
N = args.N
Re = args.Re
space_dim = args.space_dim
plot_sol = args.plot_sol
solve_type = args.solve_type

formatters = {'Time': '{:5.1f}',
              'NL Iteration': '{:5.0f}',
              'L Iteration': '{:5.1f}'}

options = solver_options.PETScOptions(solve_type=solve_type)

dual_ele = options.dual_ele
primal_ele = options.primal_ele
pcd = options.pcd
v_cycle_unassembled = options.v_cycle_unassembled

table_dict = {}
for name in schur:
    l_iterations = []
    nl_iterations = []
    time = []
    DoF = []
    boader = "#" * (len(name) + 10)
    indent = " " * 5
    print(f"  {boader}\n  {indent}{name.upper()}\n  {boader}")
    if name == "pcd":
        ns_params = options.nonlinear_solve(v_cycle_unassembled, pcd,
                                            fact_type="full")
    elif name == "dual":
        ns_params = options.nonlinear_solve(v_cycle_unassembled, dual_ele,
                                            fact_type="full")
    elif name == "primal":
        ns_params = options.nonlinear_solve(primal_ele, L2_inner)

    pprint.pprint(ns_params)
    for i in range(N):
        # i += 1
        appctx = {"velocity_space": 0, "Re": Re}
        if space_dim == "2D":
            problem = BFS_problem_2D(i, Re=Re)
        elif space_dim == "3D":
            problem = BFS_problem_3D(i, Re=Re)
        else:
            raise ValueError("space_dim variable needs to be 2D or 3D, "
                             f"currently give {space_dim}")
        ns_solver = solver.Solver(problem=problem,
                                  params=ns_params,
                                  appctx=appctx)
        output_dict = ns_solver.solve(plot_sol)

        time.append(output_dict["time"])
        l_iterations.append(output_dict["linear_iter"])
        nl_iterations.append(output_dict["nonlinear_iter"])
        DoF.append(output_dict["W_dim"])

    table_dict[name] = {"Time": time,
                        "NL Iteration": nl_iterations,
                        "L Iteration": l_iterations}

columns = [u'Time', u'L Iteration', u'NL Iteration']
table = utils.combine_tables(table_dict, DoF, columns, formatters)
name = f"navier_stokes_bfs_{space_dim}_Re={Re}.tex"
print(table)
table.to_latex(name)
