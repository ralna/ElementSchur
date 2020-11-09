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
        mesh = problem.Z.mesh()
        N1 = FunctionSpace(mesh, "N1curl", 1)
        HDiv = FunctionSpace(mesh, "BDM", 1)
        W = MixedFunctionSpace([N1, HDiv])

        sigma, u = TrialFunctions(W)
        tau, v = TestFunctions(W)
        dim = mesh.geometric_dimension()
        if dim == 3:
            a = ((1. / problem.Re) * dot(tau, sigma)
                 - inner(curl(tau), u)
                 + inner(curl(sigma), v)
                 + dot(div(v), div(u)) + dot(v, u)) * dx
        else:
            a = (dot(tau, sigma)
                 + curl(tau) * u[0]
                 + curl(tau) * u[1]
                 + curl(sigma) * v[0]
                 + curl(sigma) * v[1]
                 + problem.Re * (dot(div(v), div(u)) + dot(v, u))) * dx
        return a
