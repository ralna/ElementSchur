from firedrake.preconditioners.base import PCBase
from firedrake.petsc import PETSc
from firedrake import *
from firedrake.assemble import allocate_matrix, create_assembly_callable


class MYPC(PCBase):

    needs_python_pmat = True

    def initialize(self, pc):

        options_prefix = pc.getOptionsPrefix() + "custom_" + self.prefix + "_"

        _, P = pc.getOperators()
        context = P.getPythonContext()
        appctx = self.get_appctx(pc)
        test, trial = context.a.arguments()
        appctx["V"] = test.function_space()
        if "state" in appctx and "velocity_space" in appctx:
            u_k = split(appctx["state"])[appctx["velocity_space"]]

        a = appctx["a"] if "a" in appctx else self.schur_element_blocks(appctx)
        S = self.schur_assemble(appctx)

        if not isinstance(S, list):
            S = [S]

        self.ksp = []
        bcs = appctx["bcs"] if "bcs" in appctx else context.row_bcs

        for x, s in enumerate(S):
            prefix = "{}{}_".format(options_prefix, x) \
                if len(S) > 1 else options_prefix
            print(prefix)
            A = assemble(s, bcs=bcs,
                         form_compiler_parameters=context.fc_params,
                         mat_type="aij", options_prefix=prefix)

            ksp = PETSc.KSP().create(comm=pc.comm)
            ksp.incrementTabLevel(1, parent=pc)
            ksp.setOperators(A.petscmat)
            ksp.setOptionsPrefix(options_prefix)
            ksp.setFromOptions()
            ksp.setUp()
            self.ksp.append(ksp)

    def schur_assemble(self, appctx):
        return a

    def schur_element_blocks(self, appctx):
        pass

    def update(self, pc):
        pass

    def apply(self, pc, x, y):
        for ksp in self.ksp:
            z = y.duplicate()
            ksp.solve(x, z)
            y.array += z.array

    applyTranspose = apply

    def view(self, pc, viewer=None):
        super(MYPC, self).view(pc, viewer)
        viewer.printfASCII(f"KSP solver for {self.prefix} preconditioner\n")
        for ksp in self.ksp:
            ksp.view()


class DualElementSchur(MYPC):

    def __init__(self):
        super(DualElementSchur, self).__init__()
        self.prefix = "dual"

    def schur_assemble(self, appctx):
        AA = Tensor(appctx["a"])
        A = AA.blocks
        return A[1, 0] * A[0, 0].inv * A[0, 1]


class PrimalElementSchur(MYPC):

    def __init__(self):
        super(PrimalElementSchur, self).__init__()
        self.prefix = "primal"

    def schur_assemble(self, appctx):
        AA = Tensor(appctx["a"])
        A = AA.blocks
        return A[0, 0] + A[0, 1] * A[1, 1].inv * A[1, 0]


class PrimalSchurInvElement(MYPC):

    def __init__(self):
        super(PrimalSchurInvElement, self).__init__()
        self.prefix = "primal_dual"

    def schur_assemble(self, appctx):
        AA = Tensor(a)
        A = AA.blocks
        return [-A[1, 0] * A[0, 0].inv * A[0, 1], -A[1, 1]]


class HcurlInner(MYPC):

    def __init__(self):
        super(HcurlInner, self).__init__()
        self.prefix = "hcurl_inner"

    def schur_assemble(self, appctx):
        V = appctx["V"]
        scale_Hcurl = appctx["scale_Hcurl"] if "scale_Hcurl" in appctx else 1

        u = TrialFunction(V)
        v = TestFunction(V)
        a = scale_Hcurl * inner(curl(u), curl(v)) * dx + inner(u, v) * dx
        return a


class HdivInner(MYPC):

    def __init__(self):
        super(HdivInner, self).__init__()
        self.prefix = "hdiv_inner"

    def schur_assemble(self, appctx):
        V = appctx["V"]
        scale_Hdiv = appctx["scale_Hdiv"] if "scale_Hdiv" in appctx else 1

        u = TrialFunction(V)
        v = TestFunction(V)
        a = scale_Hdiv * inner(div(u), div(v)) * dx + inner(u, v) * dx
        return a


class H1Inner(MYPC):

    def __init__(self):
        super(H1Inner, self).__init__()
        self.prefix = "h1_inner"

    def schur_assemble(self, appctx):
        V = appctx["V"]
        scale_l2 = appctx["scale_H1"] if "scale_l2" in appctx else 1

        u = TrialFunction(V)
        v = TestFunction(V)
        a = scale_l2 * inner(grad(u), grad(v)) * dx + inner(u, v) * dx
        return a


class H1SemiInner(MYPC):

    def __init__(self):
        super(H1SemiInner, self).__init__()
        self.prefix = "h1_semi_inner"

    def schur_assemble(self, appctx):
        V = appctx["V"]
        scale = appctx["scale_h1_semi"] if "scale_h1_semi" in appctx else 1
        u = TrialFunction(V)
        v = TestFunction(V)
        a = scale * inner(grad(u), grad(v)) * dx
        return a


class L2Inner(MYPC):

    def __init__(self):
        super(L2Inner, self).__init__()
        self.prefix = "l2_inner"

    def schur_assemble(self, appctx):
        V = appctx["V"]
        scale = appctx["scale_l2"] if "scale_l2" in appctx else 1

        u = TrialFunction(V)
        v = TestFunction(V)
        a = scale * inner(u, v) * dx
        return a
