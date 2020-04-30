from firedrake import *

from ElementSchur.problemclass import BaseProblem
from ElementSchur.preconditioners import DualElementSchur, PrimalElementSchur


class Maxwell(BaseProblem):

    def __init__(self, n, Re=1):
        super(Maxwell, self).__init__()
        self.n = n
        self.Re = Re

    def primal_space(self, mesh):
        self.V = FunctionSpace(mesh, "N1curl", 1)
        return self.V

    def dual_space(self, mesh):
        self.Q = FunctionSpace(mesh, "CG", 1)
        return self.Q

    def mixed_space(self, V, Q):
        self.Z = V * Q
        return self.Z

    def form(self, z):
        u, p = split(z)
        v, q = TestFunctions(self.Z)
        f = self.rhs()
        a = (
            (1. / self.Re) * inner(curl(u), curl(v)) * dx
            + inner(v, grad(p)) * dx
            + inner(u, grad(q)) * dx
        )
        l = inner(f, v) * dx
        self.F = a - l
        return self.F

    def initial_guess(self, Z):
        self.z = Function(Z)
        return self.z


class MaxwellEleDual(DualElementSchur):

    def form(self, appctx, problem):
        u, p = TrialFunctions(problem.Z)
        v, q = TestFunctions(problem.Z)
        a = (
            (1. / problem.Re) * inner(curl(u), curl(v)) * dx + inner(u, v) * dx
            + inner(v, grad(p)) * dx
            + inner(u, grad(q)) * dx
        )
        return a


class MaxwellElePrimal(PrimalElementSchur):

    def form(self, appctx, problem):
        u, p = TrialFunctions(problem.Z)
        v, q = TestFunctions(problem.Z)
        scale = appctx["scale_h1_semi"] if "scale_h1_semi" in appctx else 1

        eps = 1e-6
        a = (
            (1. / problem.Re) * inner(curl(u), curl(v)) * dx
            + inner(v, grad(p)) * dx
            + inner(u, grad(q)) * dx
            + scale * inner(grad(p), grad(q)) * dx + eps * inner(p, q) * dx
        )
        return a
