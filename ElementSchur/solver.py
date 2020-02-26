from firedrake import *

import matplotlib.pylab as plt
from datetime import datetime


class Solver(object):

    def __init__(self, problem, params, schur_type="dual", appctx={}):
        self.problem = problem
        self.params = params
        self.appctx = appctx
        self.nu = self.problem.nu

        self.mesh = self.problem.mesh()
        V = self.problem.primal_space(self.mesh)
        Q = self.problem.dual_space(self.mesh)
        self.Z = self.problem.mixed_space()

        self.appctx["a"] = self.problem.linear_form(self.mesh, schur_type)
        nsp = problem.nullspace(self.Z)
        self.bcs = problem.bcs()
        self.z = Function(self.Z)
        self.F = self.problem.form(self.z)
        NL_problem = NonlinearVariationalProblem(self.F, self.z, bcs=self.bcs)
        self.solver = NonlinearVariationalSolver(NL_problem,
                                                 solver_parameters=self.params,
                                                 nullspace=nsp,
                                                 appctx=self.appctx)

    def linear_solve(self, plot_sol=None):
        print(f"\n\nTotal degrees of freedom {self.Z.dim()}")
        print(f"Solving for nu = {self.nu}")

        try:
            start = datetime.now()
            self.solver.solve()
            end = datetime.now()
            time = (end - start).total_seconds()
            nonlin_it = self.solver.snes.getIterationNumber()
            lin_it = self.solver.snes.getLinearSolveIterations() / nonlin_it
            print(f"CONVERGED -- nonlin iter {nonlin_it}, lin iters {lin_it}, "
                  f"time {time}")
        except Exception as excp:
            print("DIVERGED:", str(excp))
            lin_it = 0
            nonlin_it = 0
            time = 0

        F = assemble(self.F)
        for bc in self.bcs:
            bc.zero(F)
        with F.dat.vec_ro as v:
            print(f"Residual: {v.norm()}")
        if plot_sol:
            u = self.z.split()[0]
            plot(u)
            plt.show()
        info_dict = {
            "W_dim": self.Z.dim(),
            "nu": self.nu,
            "linear_iter": lin_it,
            "time": time,
        }
        return info_dict
