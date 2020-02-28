class PETScOptions(object):
    """docstring for PETScOptions"""

    def __init__(self, solve_type="iterative"):

        self.direct = {"ksp_type": "preonly",
                       "pc_type": "lu",
                       "pc_factor_mat_solver_type": "mumps"}

        self.direct_unassembled = {"ksp_type": "preonly",
                                   "pc_type": "python",
                                   "pc_python_type": "firedrake.AssembledPC",
                                   "assembled_pc_type": "lu",
                                   "assembled_pc_factor_mat_solver_type": "mumps"}
        self.ilu = {"ksp_type": "gmres",
                    "pc_type": "python",
                    "pc_python_type": "firedrake.AssembledPC",
                    "assembled_pc_type": "ilu"}
        self.v_cycle_unassembled = {"ksp_type": "preonly",
                                    "pc_type": "python",
                                    "pc_python_type": "firedrake.AssembledPC",
                                    "assembled_pc_type": "hypre"}

        self.v_cycle = {"ksp_type": "preonly",
                        "pc_type": "hypre"}

        solver = self.v_cycle if solve_type == "iterative" else self.direct

        self.H1_inner = {"ksp_type": "preonly",
                         "pc_type": "python",
                         "ksp_max_it": 1,
                         "pc_python_type": "ElementSchur.preconditioners.H1SemiInner",
                         "custom_h1_semi_inner": solver}
        self.L2_inner = {"ksp_type": "preonly",
                         "pc_type": "python",
                         "ksp_max_it": 1,
                         "pc_python_type": "ElementSchur.preconditioners.L2Inner",
                         "custom_l2_inner": solver}
        self.Hcurl_inner = {"ksp_type": "preonly",
                            "pc_type": "python",
                            "ksp_max_it": 1,
                            "pc_python_type": "ElementSchur.preconditioners.HcurlInner",
                            "custom_hcurl_inner": self.direct}

        self.dual_ele = {"ksp_type": "preonly",
                         "pc_type": "python",
                         "pc_python_type": "ElementSchur.preconditioners.DualElementSchur",
                         "custom_dual": solver}

        self.primal_ele = {"ksp_type": "preonly",
                           "pc_type": "python",
                           "ksp_max_it": 1,
                           "pc_python_type":
                           "ElementSchur.preconditioners.PrimalElementSchur",
                           "custom_primal": self.direct}
        self.pcd = {
            "ksp_type": "preonly",
            "pc_type": "python",
            "pc_python_type": "firedrake.PCDPC",
            "pcd_Mp": solver,
            "pcd_Kp": solver,
            "pcd_Fp_mat_type": "matfree"
        }

    def linear_solve(self, primal, dual, fieldsplit_type="schur",
                     fact_type="diag"):
        outer = {
            "snes_type": "ksponly",
            "mat_type": "matfree",
            "ksp_type": "gmres",
            "ksp_atol": 1e-6,
            "ksp_rtol": 1e-8,
            "ksp_max_it": 500,
            # "ksp_monitor_true_residual": None,
            "pc_type": "fieldsplit",
            "pc_fieldsplit_type": fieldsplit_type,
            "pc_fieldsplit_schur_fact_type": fact_type,
            "fieldsplit_0": primal,
            "fieldsplit_1": dual
        }
        return outer

    def nonlinear_solve(self, primal, dual, fieldsplit_type="schur",
                        fact_type="diag"):
        outer = {"snes_type": "newtonls",
                 "snes_linesearch_type": "l2",
                 "snes_linesearch_maxstep": 1.0,
                 "snes_monitor": None,
                 "snes_linesearch_monitor": None,
                 "snes_atol": 1.0e-6,
                 "snes_rtol": 1.0e-8,
                 "snes_stol": 0.0,
                 "snes_max_L_solve_fail": 10,
                 "snes_max_it": 10,
                 "mat_type": "matfree",
                 "ksp_type": "gmres",
                 "ksp_atol": 1e-6,
                 "ksp_rtol": 1e-8,
                 "ksp_max_it": 500,
                 # "ksp_monitor_true_residual": None,
                 "pc_type": "fieldsplit",
                 "pc_fieldsplit_type": fieldsplit_type,
                 "pc_fieldsplit_schur_fact_type": fact_type,
                 "fieldsplit_0": primal,
                 "fieldsplit_1": dual
                 }
        return outer
