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
        self.dual_ele = {"ksp_type": "preonly",
                         "pc_type": "python",
                         "pc_python_type": "ElementSchur.preconditioners.DualElementSchur",
                         "custom_dual": solver}

    def linear_solve(self, primal, dual, fieldsplit_type="schur",
                     fact_type="diag"):
        outer = {
            "mat_type": "matfree",
            "ksp_type": "fgmres",
            "ksp_atol": 1e-8,
            "ksp_rtol": 1e-8,
            "ksp_max_it": 500,
            "pc_type": "fieldsplit",
            "pc_fieldsplit_type": fieldsplit_type,
            "pc_fieldsplit_schur_fact_type": fact_type,
            "fieldsplit_0": primal,
            "fieldsplit_1": dual
        }
        return outer
