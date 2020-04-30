import argparse
from firedrake import *
import pprint

from ElementSchur import solver, maxwell, solver_options, utils

parser = argparse.ArgumentParser(
    description="An implementation of the Mixed Maxwell's equations.",
    add_help=True)
parser.add_argument('-s', '--schur', nargs='+',
                    default=['dual', 'primal', 'riesz'],
                    help='Schur complement approximation type (default '
                    'dual primal riesz')
parser.add_argument('-N', '--N', type=int, required=True,
                    help='Number of mesh levels')
parser.add_argument('-Re', '--Re', type=float, default=1,
                    help='Viscosity of the fluid (default 1)')
parser.add_argument('-d', '--space-dim', type=str, default="2D",
                    help='Spacial dimension of the problem (default 2D)')
parser.add_argument('--plot-sol', type=str, default=False,
                    help='Plot solution (default False)')
parser.add_argument('--solve-type', type=str, default="iterative",
                    help='Defines the solver type to be direct or iterative '
                         '(default iterative)')
args, _ = parser.parse_known_args()


class Maxwell_2D(maxwell.Maxwell):

    def __init__(self, n, Re=1):
        super(Maxwell_2D, self).__init__(n, Re)
        self.name = "maxwell_2d"

    def initial_conditions(self):
        (x, y) = SpatialCoordinate(self.Z.mesh())
        self.u0 = curl(exp(x) * cos(x + y))
        self.p0 = x * y * cos(x + y)
        return self.u0, self.p0

    def mesh_domain(self):
        mesh = UnitSquareMesh(self.n, self.n)
        return mesh

    def nullspace(self, Z):
        return None

    def bcs(self):
        bcs = [DirichletBC(self.Z.sub(0), self.u0, "on_boundary"),
               DirichletBC(self.Z.sub(1), self.p0, "on_boundary")]
        return bcs

    def rhs(self):
        u0, p0 = self.initial_conditions()
        f = (1. / self.Re) * curl(curl(u0)) + grad(p0)
        return f


class Maxwell_3D(maxwell.Maxwell):

    def __init__(self, n, Re=1):
        super(Maxwell_3D, self).__init__(n, Re)
        self.name = "maxwell_3d"

    def initial_conditions(self):
        (x, y, z) = SpatialCoordinate(self.Z.mesh())
        self.u0 = curl(as_vector([exp(x) * sin(y), x * z * y, cos(x * y)]))
        self.p0 = x * y
        return self.u0, self.p0

    def mesh_domain(self):
        mesh = UnitCubeMesh(self.n, self.n, self.n)
        return mesh

    def nullspace(self, Z):
        return None

    def bcs(self):
        bcs = [DirichletBC(self.Z.sub(0), self.u0, "on_boundary"),
               DirichletBC(self.Z.sub(1), self.p0, "on_boundary")]
        return bcs

    def rhs(self):
        u0, p0 = self.initial_conditions()
        f = (1. / self.Re) * curl(curl(u0)) + grad(p0)
        return f


schur = args.schur
N = args.N
Re = args.Re
space_dim = args.space_dim
plot_sol = args.plot_sol
solve_type = args.solve_type

formatters = {'Time': '{:5.1f}',
              'Iteration': '{:5.0f}'}

options = solver_options.PETScOptions(solve_type=solve_type)

dual_ele = options.custom_pc_amg("ElementSchur.maxwell.MaxwellEleDual", "dual")
primal_ele = options.custom_pc_direct(
    "ElementSchur.maxwell.MaxwellElePrimal", "primal")

Hcurl_inner = options.custom_pc_direct(
    "ElementSchur.preconditioners.HcurlInner", "hcurl")
H1_inner = options.custom_pc_amg(
    "ElementSchur.preconditioners.H1SemiInner", "h1_semi")

table_dict = {}
for name in schur:
    iterations = []
    time = []
    DoF = []
    boader = "#" * (len(name) + 10)
    indent = " " * 5
    print(f"  {boader}\n  {indent}{name.upper()}\n  {boader}")

    appctx = {"scale_Hcurl": 1. / Re}
    if name == "riesz":
        params = options.linear_solve(Hcurl_inner, H1_inner)
    elif name == "dual":
        params = options.linear_solve(Hcurl_inner, dual_ele)
    elif name == "primal":
        params = options.linear_solve(primal_ele, H1_inner)
        appctx["scale_h1_semi"] = 1

    pprint.pprint(params)
    for i in range(N):

        n = 2**(i + 2)

        if space_dim == "2D":
            problem = Maxwell_2D(n, Re=Re)
        elif space_dim == "3D":
            problem = Maxwell_3D(n, Re=Re)
        else:
            raise ValueError("space_dim variable needs to be 2D or 3D, "
                             f"currently give {space_dim}")
        maxwell_solver = solver.Solver(problem=problem,
                                       params=params,
                                       appctx=appctx)
        output_dict = maxwell_solver.solve(plot_sol)

        time.append(output_dict["time"])
        iterations.append(output_dict["linear_iter"])
        DoF.append(output_dict["W_dim"])

        table_dict[name] = {"Time": time, "Iteration": iterations}

columns = [u'Time', u'Iteration']
table = utils.combine_tables(table_dict, DoF, columns, formatters)
name = f"maxwell_{space_dim}_Re={Re}.tex"
print(table)
table.to_latex(name)
