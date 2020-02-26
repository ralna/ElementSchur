from firedrake import *


class BaseProblem(object):

    def mesh(self, n):
        raise NotImplementedError

    def primal_space(self):
        raise NotImplementedError

    def dual_space(self):
        raise NotImplementedError

    def form(self):
        raise NotImplementedError

    def bcs(self, Z):
        raise NotImplementedError

    def has_nullspace(self):
        raise NotImplementedError

    def nullspace(self, Z):
        raise NotImplementedError

    def mesh_size(self, u):
        return CellSize(u.ufl_domain())

    def rhs(self, Z):
        return None
