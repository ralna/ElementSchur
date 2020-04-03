from firedrake import *

import os
import matplotlib.pylab as plt
from datetime import datetime
import json


class Solver(object):

    def __init__(self, problem, params, appctx={}, z=None):
        self.problem = problem
        self.params = params
        self.appctx = appctx
        self.Re = self.problem.Re

        self.mesh = self.problem.mesh_domain()
        self.V = self.problem.primal_space(self.mesh)
        self.Q = self.problem.dual_space(self.mesh)
        self.Z = self.problem.mixed_space(self.V, self.Q)
        self.z = problem.initial_guess(self.Z)

        self.F = self.problem.form(self.z)
        self.appctx["problem"] = self.problem

        nsp = problem.nullspace(self.Z)
        self.bcs = problem.bcs()
        NL_problem = NonlinearVariationalProblem(self.F, self.z, bcs=self.bcs)
        self.solver = NonlinearVariationalSolver(NL_problem,
                                                 solver_parameters=self.params,
                                                 nullspace=nsp,
                                                 appctx=self.appctx)
        print(f"\n\nTotal degrees of freedom {self.Z.dim()}")

    def solve(self, plot_sol=None):
        print(f"Solving for Re = {self.Re}")

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
        u = self.z.split()[0]
        if plot_sol:
            print(u)
            quiver(u)
            plt.show()
        filename = f"{self.problem.name}_Re={self.Re}_n={self.Z.dim()}"
        info_dict = {
            "W_dim": self.Z.dim(),
            "Re": self.Re,
            "linear_iter": lin_it,
            "nonlinear_iter": nonlin_it,
            "time": time
        }
        self.save_sol(u, info_dict, filename)
        return info_dict

    def save_sol(self, u, info_dict, filename):
        wd = os.getcwd()
        results_dir = os.path.join(wd, "results", f"{self.problem.name}")
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        filename_path_fig = os.path.join(results_dir, f"{filename}.pvd")
        filename_path_dict = os.path.join(results_dir, f"{filename}.json")
        outfile = File(filename_path_fig)
        outfile.write(u)
        with open(filename_path_dict, "w") as f:
            json.dump(info_dict, f)
