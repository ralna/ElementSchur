from firedrake.preconditioners.base import PCBase
from firedrake.petsc import PETSc
from firedrake import *
from firedrake.assemble import allocate_matrix, create_assembly_callable


class MYPC(PCBase):

    needs_python_pmat = True

    def initialize(self, pc):
        self.options_prefix = f"{pc.getOptionsPrefix()}custom_{self.prefix}_"
        print(self.options_prefix)
        _, P = pc.getOperators()
        context = P.getPythonContext()
        appctx = self.get_appctx(pc)
        test, trial = context.a.arguments()
        appctx["V"] = test.function_space()
        if "state" in appctx and "velocity_space" in appctx:
            appctx["u_k"] = appctx["state"].split()[appctx["velocity_space"]]

        problem = appctx["problem"]
        linear_form = self.form(appctx, problem)
        self.a = self.assemble_ele_schur(linear_form)

        bcs = appctx["bcs"] if "bcs" in appctx else context.row_bcs

        self.A = allocate_matrix(
            self.a, bcs=bcs, form_compiler_parameters=context.fc_params,
            mat_type="aij", options_prefix=self.options_prefix)
        self._assemble_form = create_assembly_callable(
            self.a, tensor=self.A, bcs=bcs,
            form_compiler_parameters=context.fc_params, mat_type="aij")

        self._assemble_form()
        self.ksp = self.setup_ksp(self.A.petscmat, pc)

    def form(self, appctx, problem):
        raise NotImplementedError

    def assemble_ele_schur(self, a):
        return a

    def update(self, pc):
        self._assemble_form()
        self.ksp = self.setup_ksp(self.A.petscmat, pc)

    def apply(self, pc, x, y):
        self.ksp.solve(x, y)

    applyTranspose = apply

    def view(self, pc, viewer=None):
        super(MYPC, self).view(pc, viewer)
        viewer.printfASCII(f"KSP solver for {self.prefix} preconditioner\n")
        ksp.view()

    def setup_ksp(self, A, pc):
        ksp = PETSc.KSP().create(comm=pc.comm)
        ksp.incrementTabLevel(1, parent=pc)
        ksp.setOperators(A)
        ksp.setOptionsPrefix(self.options_prefix)
        ksp.setFromOptions()
        ksp.setUp()
        return ksp

class PrimalDual(PCBase):

    needs_python_pmat = True

    def initialize(self, pc):
        self.options_prefix = f"{pc.getOptionsPrefix()}custom_primal_dual_"
        print(self.options_prefix)
        _, P = pc.getOperators()
        context = P.getPythonContext()
        appctx = self.get_appctx(pc)
        test, trial = context.a.arguments()
        appctx["V"] = test.function_space()
        if "state" in appctx and "velocity_space" in appctx:
            appctx["u_k"] = appctx["state"].split()[appctx["velocity_space"]]

        problem = appctx["problem"]
        linear_form = self.form(appctx, problem)

        self.a = self.assemble_ele_schur(linear_form)
        AA = Tensor(self.a)
        A = AA.blocks
        schur = A[1, 0] * A[0, 0].inv * A[0, 1]
        Q = A[1, 1]

        bcs = appctx["bcs"] if "bcs" in appctx else context.row_bcs
        self.A_schur = allocate_matrix(
            schur, bcs=bcs, form_compiler_parameters=context.fc_params,
            mat_type="aij", options_prefix=self.options_prefix)
        self._assemble_form_schur = create_assembly_callable(
            schur, tensor=self.A_schur, bcs=bcs,
            form_compiler_parameters=context.fc_params, mat_type="aij")

        self._assemble_form_schur()
        self.ksp_schur = self.setup_ksp(self.A_schur.petscmat, pc)

        self.A_Q = allocate_matrix(
            Q, bcs=bcs, form_compiler_parameters=context.fc_params,
            mat_type="aij", options_prefix=self.options_prefix)
        self._assemble_form_Q = create_assembly_callable(
            Q, tensor=self.A_Q, bcs=bcs,
            form_compiler_parameters=context.fc_params, mat_type="aij")

        self._assemble_form_Q()
        self.ksp_Q = self.setup_ksp(self.A_Q.petscmat, pc)

    def form(self, appctx, problem):
        raise NotImplementedError

    def assemble_ele_schur(self, a):
        return a

    def update(self, pc):
        self._assemble_form_schur()
        self.ksp_schur = self.setup_ksp(self.A_schur.petscmat, pc)
        # self._assemble_form_Q()
        # self.ksp_q = self.setup_ksp(self.A_Q.petscmat, pc)

    def apply(self, pc, x, y):
        z1 = x.duplicate()
        z2 = x.duplicate()
        self.ksp_schur.solve(x, z1)
        self.ksp_Q.solve(x, z2)
        y.array = (z1.array - z2.array)

    applyTranspose = apply

    def view(self, pc, viewer=None):
        super(MYPC, self).view(pc, viewer)
        viewer.printfASCII(f"KSP solver for {self.prefix} preconditioner\n")
        ksp.view()

    def setup_ksp(self, A, pc):
        ksp = PETSc.KSP().create(comm=pc.comm)
        ksp.incrementTabLevel(1, parent=pc)
        ksp.setOperators(A)
        ksp.setOptionsPrefix(self.options_prefix)
        ksp.setFromOptions()
        ksp.setUp()
        return ksp


class DualElementSchur(MYPC):

    def __init__(self):
        super(DualElementSchur, self).__init__()
        self.prefix = "dual"

    def assemble_ele_schur(self, linear_form):
        AA = Tensor(linear_form)
        A = AA.blocks
        return A[1, 0] * A[0, 0].inv * A[0, 1]


class PrimalElementSchur(MYPC):

    def __init__(self):
        super(PrimalElementSchur, self).__init__()
        self.prefix = "primal"

    def assemble_ele_schur(self, linear_form):
        AA = Tensor(linear_form)
        A = AA.blocks
        return A[0, 0] + A[0, 1] * A[1, 1].inv * A[1, 0]


class HcurlInner(MYPC):

    def __init__(self):
        super(HcurlInner, self).__init__()
        self.prefix = "hcurl"

    def form(self, appctx, problem):
        V = appctx["V"]
        scale_Hcurl = appctx["scale_Hcurl"] if "scale_Hcurl" in appctx else 1

        u = TrialFunction(V)
        v = TestFunction(V)
        a = scale_Hcurl * inner(curl(u), curl(v)) * dx + inner(u, v) * dx
        return a


class HdivInner(MYPC):

    def __init__(self):
        super(HdivInner, self).__init__()
        self.prefix = "hdiv"

    def form(self, appctx, problem):
        V = appctx["V"]
        scale_Hdiv = appctx["scale_Hdiv"] if "scale_Hdiv" in appctx else 1

        u = TrialFunction(V)
        v = TestFunction(V)
        a = scale_Hdiv * inner(div(u), div(v)) * dx + inner(u, v) * dx
        return a


class H1Inner(MYPC):

    def __init__(self):
        super(H1Inner, self).__init__()
        self.prefix = "h1"

    def form(self, appctx, problem):
        V = appctx["V"]
        scale_l2 = appctx["scale_H1"] if "scale_l2" in appctx else 1

        u = TrialFunction(V)
        v = TestFunction(V)
        a = scale_l2 * inner(grad(u), grad(v)) * dx + inner(u, v) * dx
        return a


class H1SemiInner(MYPC):

    def __init__(self):
        super(H1SemiInner, self).__init__()
        self.prefix = "h1_semi"

    def form(self, appctx, problem):
        V = appctx["V"]
        scale = appctx["scale_h1_semi"] if "scale_h1_semi" in appctx else 1
        u = TrialFunction(V)
        v = TestFunction(V)
        a = scale * inner(grad(u), grad(v)) * dx
        return a


class L2Inner(MYPC):

    def __init__(self):
        super(L2Inner, self).__init__()
        self.prefix = "l2"

    def form(self, appctx, problem):
        V = appctx["V"]
        scale = appctx["scale_l2"] if "scale_l2" in appctx else 1

        u = TrialFunction(V)
        v = TestFunction(V)
        a = scale * inner(u, v) * dx
        return a
