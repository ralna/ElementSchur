from firedrake import *


class StokesSolver(object):

    def __init__(self, problem, nu):
        self.problem = problem

        self.nu = nu

        self.mesh = self.problem.mesh()
        V = self.problem.primal_space(mesh)
        Q = self.problem.dual_space(mesh)
        Z = self.problem.mixed_space([V, Q])

        z = Function(Z)
        z.split()[0].rename("Velocity")
        z.split()[1].rename("Pressure")

        (v, q) = split(TestFunction(Z))

        nsp = problem.nullspace(Z)
        bcs = problem.bcs(Z)

        problem = NonlinearVariationalProblem(self.problem.form(z), z, bcs=bcs)
        solver = NonlinearVariationalSolver(
            problem, solver_parameters=params, nullspace=nsp, options_prefix="ns_")


    # def solve(self, re):
    #     self.message(GREEN % ("Solving for Re = %s" % re))
    #     if re == 0:
    #         z_stokes, info_dict = solve_stokes(self.mesh, self.problem, self.Z)
    #         self.z.split()[0].interpolate(z_stokes.split()[0])
    #         self.z.split()[1].interpolate(z_stokes.split()[1])
    #         return (self.z, info_dict)

    #     self.nu.assign(self.char_L * self.char_U / re)
    #     self.SUPG.update(self.z.split()[0])
    #     start = datetime.now()
    #     with dmhooks.ctx_coarsener(self.z.function_space(), coarsen):
    #         self.solver.solve()
    #     end = datetime.now()

    #     if self.check_nograddiv_residual:
    #         F_ngd = assemble(self.F_nograddiv)
    #         for bc in self.bcs:
    #             bc.zero(F_ngd)
    #         F = assemble(self.F)
    #         for bc in self.bcs:
    #             bc.zero(F)
    #         with F_ngd.dat.vec_ro as v_ngd, F.dat.vec_ro as v:
    #             self.message(
    #                 BLUE % ("Residual without grad-div term: %.14e" % v_ngd.norm()))
    #             self.message(
    #                 BLUE % ("Residual with grad-div term:    %.14e" % v.norm()))
    #     Re_linear_its = self.solver.snes.getLinearSolveIterations()
    #     Re_nonlinear_its = self.solver.snes.getIterationNumber()
    #     Re_time = (end - start).total_seconds() / 60
    #     self.message(GREEN % ("Time taken: %.2f min in %d iterations (%.2f Krylov iters per Newton step)" % (
    #         Re_time, Re_linear_its, Re_linear_its / float(Re_nonlinear_its))))
    #     info_dict = {
    #         "Re": re,
    #         "nu": self.nu.values()[0],
    #         "linear_iter": Re_linear_its,
    #         "nonlinear_iter": Re_nonlinear_its,
    #         "time": Re_time,
    #     }
    #     return (self.z, info_dict)

    # def setup_schur_complement(self, snes):
    #     # Set custom Schur complement approximation
    #     mass_test = TestFunction(self.Q)
    #     inv_mass = assemble((mass_test / (CellVolume(self.Q.mesh())**2)) * dx)

    #     Re = self.Re
    #     gamma = self.gamma

    #     class SchurInvApprox(object):
    #         def mult(self, mat, x, y):
    #             with inv_mass.dat.vec_ro as w:
    #                 y.pointwiseMult(x, w)
    #             y.scale(-float(1.0 / Re + gamma))

    #     schur = PETSc.Mat()
    #     lSize = inv_mass.vector().local_size()
    #     gSize = inv_mass.vector().size()
    #     schur.createPython(((lSize, gSize), (lSize, gSize)), SchurInvApprox())
    #     schur.setUp()
    #     snes.ksp.pc.setFieldSplitSchurPreType(PETSc.PC.SchurPreType.USER,
    #                                           schur)

    # def get_parameters(self):
    #     simple = {
    #         "snes_type": "newtonls",
    #         "snes_linesearch_type": "l2",
    #         "snes_linesearch_maxstep": 1.0,
    #         "snes_monitor": None,
    #         "snes_linesearch_monitor": None,
    #         "snes_rtol": 1.0e-10,
    #         "snes_atol": 1.0e-8,
    #         "snes_stol": 0.0,
    #         "snes_max_linear_solve_fail": 10,
    #         "mat_type": "nest",
    #         "ksp_type": "fgmres",
    #         "ksp_rtol": 1.0e-6,
    #         "ksp_atol": 1.0e-10,
    #         "ksp_max_it": 10000,
    #         "ksp_monitor": None,
    #         "ksp_converged_reason": None,
    #         "pc_type": "fieldsplit",
    #         "pc_fieldsplit_type": "schur",
    #         "pc_fieldsplit_schur_factorization_type": "full",
    #         "pc_fieldsplit_schur_precondition": "selfp",
    #         "fieldsplit_0": {
    #             "ksp_type": "richardson",
    #             "ksp_richardson_self_scale": False,
    #             "ksp_max_it": 1,
    #             "ksp_norm_type": "none",
    #             "ksp_convergence_test": "skip",
    #             "pc_type": "ml",
    #             "pc_mg_cycle_type": "v",
    #             "pc_mg_type": "full",
    #         },
    #         "fieldsplit_1": {
    #             "ksp_type": "preonly",
    #             "pc_type": "ml",
    #         },
    #         "fieldsplit_1_upper_ksp_type": "preonly",
    #         "fieldsplit_1_upper_pc_type": "jacobi",
    #     }
    #     if self.simple:
    #         parameters["default_sub_matrix_type"] = "aij"
    #         return simple

    #     size = self.mesh.mpi_comm().size
    #     if size > 24:
    #         telescope_factor = round(size / 24.0)
    #     else:
    #         telescope_factor = 1

    #     ostar = {
    #         "ksp_type": "fgmres",
    #         "ksp_norm_type": "none",
    #         "ksp_convergence_test": "skip",
    #         "ksp_max_it": 10 if self.tdim > 2 else 6,
    #         "pc_type": "python",
    #         "pc_python_type": "firedrake.PatchPC",
    #         "patch_pc_patch_save_operators": True,
    #         "patch_pc_patch_partition_of_unity": False,
    #         "patch_pc_patch_multiplicative": False,
    #         "patch_pc_patch_sub_mat_type": "seqaij" if self.tdim > 2 else "seqdense",
    #         "patch_pc_patch_construct_type": "star",
    #         "patch_pc_patch_construct_dim": 0,
    #         "patch_pc_patch_symmetrise_sweep": False,
    #         "patch_pc_patch_precompute_element_tensors": True,
    #         "patch_sub_ksp_type": "preonly",
    #         "patch_sub_pc_type": "lu",
    #     }

    #     fieldsplit_0_lu = {
    #         "ksp_type": "preonly",
    #         "ksp_max_it": 1,
    #         "pc_type": "lu",
    #         "pc_factor_mat_solver_package": "mumps",
    #         "pc_factor_mat_solver_type": "mumps",
    #     }

    #     mg_levels_solver = ostar
    #     fieldsplit_0_mg_full = {
    #         "ksp_type": "richardson",
    #         "ksp_richardson_self_scale": False,
    #         "ksp_max_it": 1,
    #         "ksp_norm_type": "none",
    #         "ksp_convergence_test": "skip",
    #         "pc_type": "mg",
    #         "pc_mg_cycle_type": "v",
    #         "pc_mg_type": "full",
    #         "mg_levels": mg_levels_solver,
    #         "mg_coarse_pc_type": "python",
    #         "mg_coarse_pc_python_type": "firedrake.AssembledPC",
    #         "mg_coarse_assembled": {
    #             "mat_type": "aij",
    #             "pc_type": "telescope",
    #             "pc_telescope_reduction_factor": telescope_factor,
    #             "pc_telescope_subcomm_type": "contiguous",
    #             "telescope_pc_type": "lu",
    #             "telescope_pc_factor_mat_solver_type": "superlu_dist",
    #         }
    #     }
    #     fieldsplit_0_mg_standard = {
    #         "ksp_type": "gcr",
    #         "ksp_max_it": 2,
    #         "ksp_norm_type": "none",
    #         "ksp_convergence_test": "skip",
    #         "pc_type": "mg",
    #         "pc_mg_cycle_type": "w",
    #         "pc_mg_type": "multiplicative",
    #         "mg_levels": mg_levels_solver,
    #         "mg_coarse_pc_type": "python",
    #         "mg_coarse_pc_python_type": "firedrake.AssembledPC",
    #         "mg_coarse_assembled_pc_type": "lu",
    #     }
    #     fieldsplit_0_mg = fieldsplit_0_mg_full

    #     fieldsplit_1 = {
    #         "ksp_type": "preonly",
    #         "pc_type": "mat",
    #     }
    #     use_mg = self.multigrid
    #     outer = {
    #         "snes_type": "newtonls",
    #         "snes_linesearch_type": "l2",
    #         "snes_linesearch_maxstep": 1.0,
    #         "snes_monitor": None,
    #         "snes_linesearch_monitor": None,
    #         "snes_rtol": 1.0e-10,
    #         "snes_atol": 1.0e-8,
    #         "snes_stol": 0.0,
    #         "snes_max_linear_solve_fail": 10,
    #         "mat_type": "nest",
    #         "ksp_type": "fgmres",
    #         "ksp_rtol": 1.0e-6,
    #         "ksp_atol": 1.0e-10,
    #         "ksp_max_it": 100,
    #         "ksp_monitor": None,
    #         "ksp_converged_reason": None,
    #         "pc_type": "fieldsplit",
    #         "pc_fieldsplit_type": "schur",
    #         "pc_fieldsplit_schur_factorization_type": "full",
    #         "pc_fieldsplit_schur_precondition": "user",
    #         "fieldsplit_0": fieldsplit_0_mg if use_mg else fieldsplit_0_lu,
    #         "fieldsplit_1": fieldsplit_1,
    #     }

    #     if self.tdim > 2:
    #         outer["ksp_atol"] = 1.0e-8
    #         outer["ksp_rtol"] = 1.0e-5
    #         outer["snes_atol"] = outer["ksp_atol"]

    #     return outer

    # def message(self, msg):
    #     if self.comm.rank == 0:
    #         warning(msg)

    # def load_balance(self, Z):
    #     owned_dofs = Z.dof_dset.sizes[1]
    #     comm = Z.mesh().mpi_comm()
    #     min_owned_dofs = comm.allreduce(owned_dofs, op=MPI.MIN)
    #     max_owned_dofs = comm.allreduce(owned_dofs, op=MPI.MAX)
    #     self.message(BLUE % ("Load balance: %i vs %i" %
    #                          (min_owned_dofs, max_owned_dofs)))
