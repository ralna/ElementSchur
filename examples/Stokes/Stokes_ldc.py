import argparse
from firedrake import *
import pprint

from ElementSchur import solver, stokes, solver_options, utils

parser = argparse.ArgumentParser(add_help=True)
parser.add_argument('-s', '--schur', nargs='+',
                    default=['dual', 'reisz'],
                    help='Schur complement approximation type (default '
                    'dual reisz')
parser.add_argument('-N', '--N', type=int, required=True,
                    help='Number of mesh levels')
parser.add_argument('-nu', '--nu', type=float, default=1,
                    help='Viscosity of the fluid (default 1)')
parser.add_argument('-d', '--space_dim', type=str, default="2D",
                    help='Spacial dimension of the problem (default 2D)')
parser.add_argument('--plot-sol', type=str, default=False,
                    help='Plot solution (default False)')
args, _ = parser.parse_known_args()


class LDC_problem_2D(stokes.Stokes):

    def __init__(self, n, nu=1):
        super(LDC_problem_2D, self).__init__(n, nu)

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


class LDC_problem_3D(stokes.Stokes):

    def __init__(self, n, nu=1):
        super(LDC_problem_3D, self).__init__(n, nu)

    def mesh_domain(self):
        mesh = UnitCubeMesh(self.n, self.n, self.n)
        return mesh

    def nullspace(self, Z):
        MVSB = MixedVectorSpaceBasis
        nullspace = MVSB(Z, [Z.sub(0), VectorSpaceBasis(constant=True)])
        return nullspace

    def bcs(self):
        bcs = [DirichletBC(self.Z.sub(0), Constant((1, 0, 0)), (4,)),
               DirichletBC(self.Z.sub(0), Constant((0, 0, 0)), [1, 2, 3, 5, 6])]
        return bcs

    def rhs(self):
        return Constant((0., 0., 0.))


schur = args.schur
N = args.N
nu = args.nu
space_dim = args.space_dim
plot_sol = args.plot_sol

formatters = {'Time': '{:5.1f}',
              'Iteration': '{:5.0f}'}

options = solver_options.PETScOptions()

dual_ele = options.dual_ele
primal_ele = options.primal_ele
L2_inner = options.L2_inner
H1_inner = options.H1_inner

table_dict = {}
for name in schur:
    iterations = []
    time = []
    DoF = []
    boader = "#" * (len(name) + 10)
    indent = " " * 5
    print(f"  {boader}\n  {indent}{name.upper()}\n  {boader}")
    if name == "reisz":
        params = options.linear_solve(H1_inner, L2_inner)
    elif name == "dual":
        params = options.linear_solve(H1_inner, dual_ele)
    elif name == "primal":
        params = options.linear_solve(primal_ele, L2_inner)

    pprint.pprint(params)
    for i in range(N):

        n = 2**(i + 1)
        appctx = {"scale_l2": 1. / nu, "scale_h1_semi": nu}
        if space_dim == "2D":
            problem = LDC_problem_2D(n, nu=nu)
        elif space_dim == "3D":
            problem = LDC_problem_3D(n, nu=nu)
        else:
            raise ValueError("space_dim variable needs to be 2D or 3D, "
                             f"currently give {space_dim}")
        stokes_solver = solver.Solver(problem=problem,
                                      params=params,
                                      appctx=appctx)
        output_dict = stokes_solver.solve(plot_sol)
        time.append(output_dict["time"])
        iterations.append(output_dict["linear_iter"])
        DoF.append(output_dict["W_dim"])

        table_dict[name] = {"Time": time, "Iteration": iterations}

columns = [u'Time', u'Iteration']
table = utils.combine_tables(table_dict, DoF, columns, formatters)
print(table)
