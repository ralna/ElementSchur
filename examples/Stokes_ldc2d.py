from firedrake import *
import pandas as pd

from ElementSchur import problemclass, solver, stokes


class LDC_problem(stokes.Stokes):

    def __init__(self, n, nu=1):
        super(LDC_problem, self).__init__(n, nu)

    def nullspace(self, Z):
        MVSB = MixedVectorSpaceBasis
        nullspace = MVSB(Z, [Z.sub(0), VectorSpaceBasis(constant=True)])
        return nullspace

    def bcs(self):
        bcs = [DirichletBC(self.W.sub(0), Constant((1, 0)), (4,)),
               DirichletBC(self.W.sub(0), Constant((0, 0)), [1, 2, 3])]
        return bcs


def precond_setup(a, b):
    outer = {"snes_type": "ksponly",
             "mat_type": "matfree",
             "ksp_type": "gmres",
             # "ksp_view": True,
             "ksp_atol": 1e-6,
             "ksp_rtol": 1e-10,
             # "ksp_monitor_true_residual": None,
             "ksp_max_it": 500,
             "pc_type": "fieldsplit",
             "pc_fieldsplit_type": "multiplicative",
             # "pc_fieldsplit_schur_fact_type": "diag",
             "fieldsplit_0": a,
             "fieldsplit_1": b
             }
    return outer


v_cycle_unassembled = {"ksp_type": "preonly",
                       "pc_type": "python",
                       "pc_python_type": "firedrake.AssembledPC",
                       "assembled_pc_type": "hypre"}
v_cycle = {"ksp_type": "preonly",
           "pc_type": "hypre"}

L2_inner = {"ksp_type": "preonly",
            "pc_type": "python",
            "ksp_max_it": 1,
            "pc_python_type": "ElementSchur.preconditioners.L2Inner",
            "custom_l2_inner": v_cycle}
dual_schur = {"ksp_type": "preonly",
              "pc_type": "python",
              "pc_python_type":
              "ElementSchur.preconditioners.DualElementSchur",
              "custom_dual": v_cycle}
n = 8
nu = 1

formatters = {'Time': '{:5.1f}',
              'Iteration': '{:5.0f}'}
data_frames = {}
for name, dual in zip(["mass", "ele"], [L2_inner, dual_schur]):
    iterations = []
    time = []
    DoF = []

    for i in range(7):
        n = 2**(i + 1)
        apptx = {"scale_l2": 1. / nu}
        problem = LDC_problem(n, nu=nu)
        s = solver.Solver(problem,
                          precond_setup(v_cycle_unassembled, dual),
                          apptx)
        output_dict = s.linear_solve()
        time.append(output_dict["time"])
        iterations.append(output_dict["linear_iter"])
        DoF.append(output_dict["W_dim"])

    table_dict = {"Time": time, "Iteration": iterations}

    cols = pd.MultiIndex.from_product([[name], [u'Time', u'Iteration']])
    df = pd.DataFrame(data=table_dict)

    for col_name, col in df.iteritems():
        df[col_name] = col.apply(lambda x: formatters[col_name].format(x))

    df.columns = cols
    data_frames[name] = df

concat_results = [data_frames[key] for key in data_frames.keys()]
final_table = pd.concat([pd.DataFrame(data={"DoF": DoF})] + concat_results,
                        axis=1, sort=False)
final_table["DoF"] = final_table["DoF"].apply(lambda x: f"{x:5.0f}")
print(final_table)
