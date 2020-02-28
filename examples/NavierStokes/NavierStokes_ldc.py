import argparse
from firedrake import *
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


class LDC_problem_2D(navier_stokes.NavierStokes):

    def __init__(self, n, Re=1):
        super(LDC_problem_2D, self).__init__(n, Re)
        self.name = "ns_ldc_2d"

    def mesh_domain(self):
        mesh = UnitSquareMesh(self.n, self.n)
        return mesh

    def nullspace(self, Z):
        MVSB = MixedVectorSpaceBasis
        nullspace = MVSB(Z, [Z.sub(0), VectorSpaceBasis(constant=True)])
        return nullspace

    def bcs(self):
        bcs = [DirichletBC(self.Z.sub(0), Constant((1, 0)), (4,)),
               DirichletBC(self.Z.sub(0), Constant((0, 0)), [1, 2, 3])]
        return bcs

    def rhs(self):
        return Constant((0., 0.))


class LDC_problem_3D(navier_stokes.NavierStokes):

    def __init__(self, n, Re=1):
        super(LDC_problem_3D, self).__init__(n, Re)
        self.name = "ns_ldc_3d"

    def mesh_domain(self):
        mesh = UnitCubeMesh(self.n, self.n, self.n)
        return mesh

    def nullspace(self, Z):
        MVSB = MixedVectorSpaceBasis
        nullspace = MVSB(Z, [Z.sub(0), VectorSpaceBasis(constant=True)])
        return nullspace

    def bcs(self):
        bcs = [DirichletBC(self.Z.sub(0), Constant((1, 0, 0)), (4,)),
               DirichletBC(self.Z.sub(0), Constant((0, 0, 0)),
                           [1, 2, 3, 5, 6])]
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
        n = 2**(i + 1)
        appctx = {"velocity_space": 0, "Re": Re}
        if space_dim == "2D":
            problem = LDC_problem_2D(n, Re=Re)
        elif space_dim == "3D":
            sss
            problem = LDC_problem_3D(n, Re=Re)
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
name = f"navier_stokes_ldc_{space_dim}_Re={Re}.tex"
print(table)
table.to_latex(name)
