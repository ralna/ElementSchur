from firedrake import *

from ElementSchur.problemclass import BaseProblem


class Maxwell(BaseProblem):

    def __init__(self, n, nu=1):
        self.n = n
        self.nu = nu

    def primal_space(self, mesh):
        self.V = FunctionSpace(mesh, "N1curl", 1)
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
            self.nu * inner(curl(u), curl(v)) * dx
            + inner(v, grad(p)) * dx
            + inner(u, grad(q)) * dx
        )
        l = inner(f, v) * dx
        self.F = a - l
        return self.F

    def linear_form(self, mesh, schur_type):
        u, p = TrialFunctions(self.Z)
        v, q = TestFunctions(self.Z)
        if schur_type == "dual":
            a = (
                self.nu * inner(curl(u), curl(v)) * dx + inner(u, v) * dx
                + inner(v, grad(p)) * dx
                + inner(u, grad(q)) * dx
            )
        elif schur_type == "primal":
            eps = CellSize(mesh)**2
            a = (
                self.nu * inner(curl(u), curl(v)) * dx
                + inner(v, grad(p)) * dx
                + inner(u, grad(q)) * dx
                + inner(grad(p), grad(q)) * dx + eps * inner(p, q) * dx
            )
        else:
            a = None
        return a
