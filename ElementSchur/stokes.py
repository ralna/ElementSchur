from firedrake import *

from ElementSchur.problemclass import BaseProblem


class Stokes(BaseProblem):

    def __init__(self, n, nu=1, f=Constant((0., 0.))):
        self.n = n
        self.nu = nu
        self.f = f

    def primal_space(self, mesh):
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
        a = (
            self.nu * inner(grad(u), grad(v)) * dx
            - p * div(v) * dx
            - q * div(u) * dx
        )
        l = inner(self.f, v) * dx
        self.F = a - l
        return self.F

    def linear_form(self, mesh, schur_type):
        u, p = TrialFunctions(self.Z)
        v, q = TestFunctions(self.Z)
        eps = CellSize(mesh)**2
        a = (
            self.nu * inner(grad(u), grad(v)) * dx
            + eps * inner(u, v) * dx
            - p * div(v) * dx
            - q * div(u) * dx
        )
        return a

    def bcs(self, Z):
        raise NotImplementedError

    def has_nullspace(self):
        raise NotImplementedError

    def nullspace(self, Z):
        raise NotImplementedError

    def rhs(self, Z):
        return None
