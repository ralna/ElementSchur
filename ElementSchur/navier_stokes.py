from firedrake import *

from ElementSchur.problemclass import BaseProblem


class NavierStokes(BaseProblem):

    def __init__(self, n, Re=1):
        super(NavierStokes, self).__init__()
        self.n = n
        self.Re = Re

    def primal_space(self, mesh, vel_degree_wo_bubble=1):
        self.V = VectorFunctionSpace(mesh, "CG", 2)
        return self.V

    def dual_space(self, mesh):
        self.Q = FunctionSpace(mesh, "CG", 1)
        return self.Q

    def mixed_space(self):
        self.Z = self.V * self.Q
        return self.Z

    def form(self, z):
        u, p = split(z)
        v, q = TestFunctions(self.Z)
        f = self.rhs()
        a = (
            (1. / self.Re) * inner(grad(u), grad(v)) * dx
            + inner(dot(grad(u), u), v) * dx
            - p * div(v) * dx
            - q * div(u) * dx
        )
        l = inner(f, v) * dx
        self.F = a - l
        return self.F

    def linear_form(self, mesh, schur_type):
        u, p = TrialFunctions(self.Z)
        v, q = TestFunctions(self.Z)
        u_k = Function(self.V)
        if schur_type == "dual":
            eps = CellSize(mesh)**2
            a = (
                -(1. / self.Re) * inner(grad(u), grad(v)) * dx
                - inner(dot(grad(u), u_k), v) * dx
                - eps * inner(u, v) * dx
                - p * div(v) * dx
                - q * div(u) * dx
            )
        elif schur_type == "primal":
            a = (
                (1. / self.Re) * inner(grad(u), grad(v)) * dx
                + inner(dot(grad(u), u_k), v) * dx
                - p * div(v) * dx
                - q * div(u) * dx
                + inner(q, p) * dx
            )
        else:
            a = None
        return a
