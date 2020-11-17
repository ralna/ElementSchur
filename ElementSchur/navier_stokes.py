from firedrake import *

from ElementSchur.problemclass import BaseProblem
from ElementSchur.preconditioners import DualElementSchur, \
    PrimalElementSchur, PrimalDual


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

    def mixed_space(self, V, Q):
        self.Z = V * Q
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

    def initial_guess(self, Z):
        self.z = Function(Z)
        return self.z


class NavierStokesEleDual(DualElementSchur):

    def form(self, appctx, problem):
        eps = 1e-6
        u, p = TrialFunctions(problem.Z)
        v, q = TestFunctions(problem.Z)

        a = (
            -(1. / problem.Re) * inner(grad(u), grad(v)) * dx
            - inner(dot(grad(u), appctx["u_k"]), v) * dx
            - eps * inner(u, v) * dx
            - p * div(v) * dx
            - q * div(u) * dx
        )
        return a

class NavierStokesEleDualAlt(DualElementSchur):

    def form(self, appctx, problem):
        mesh = problem.Z.mesh()
        BDM = FunctionSpace(mesh, "BDM", 2)
        DG = FunctionSpace(mesh, "CG", 1)
        W = BDM * DG

        sigma, u = TrialFunctions(W)
        tau, v = TestFunctions(W)
        a = (-(1. / problem.Re) * (inner(sigma, tau)
                                   + inner(div(sigma), div(tau)))
             - inner(dot(grad(sigma), appctx["u_k"]), tau)
             - div(tau) * u - div(sigma) * v) * dx
        return a

class NavierStokesElePrimal(PrimalElementSchur):

    def form(self, appctx, problem):
        u, p = TrialFunctions(problem.Z)
        v, q = TestFunctions(problem.Z)

        a = (
            +(1. / problem.Re) * inner(grad(u), grad(v)) * dx
            + inner(dot(grad(u), appctx["u_k"]), v) * dx
            - p * div(v) * dx
            - q * div(u) * dx
            + problem.Re * inner(p, q) * dx
        )
        return a


class NavierStokesElePrimalDual(PrimalDual):

    def form(self, appctx, problem):
        u, p = TrialFunctions(problem.Z)
        v, q = TestFunctions(problem.Z)
        eps = 1e-6

        a = (
            -(1. / problem.Re) * inner(grad(u), grad(v)) * dx
            - inner(dot(grad(u), appctx["u_k"]), v) * dx
            - eps * inner(u, v) * dx
            - p * div(v) * dx
            - q * div(u) * dx
            + problem.Re * inner(p, q) * dx
        )
        return a
