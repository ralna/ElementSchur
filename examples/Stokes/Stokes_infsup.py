import numpy as np
import pandas as pd
import scipy as sp

try:
    from dolfin import *
except ImportError:
    s_type = "firedrake"

try:
    from firedrake import *
    from firedrake.assemble import allocate_matrix, create_assembly_callable
except ImportError:
    s_type = "fenics"

N = [2, 4, 8, 16]
Re = 100.0
dual_min = []
dual_max = []
primal_min = []
primal_max = []
cell_num = []
for n in N:
    mesh = UnitSquareMesh(n, n)

    order = 1
    H1 = VectorFunctionSpace(mesh, "CG", order + 1)
    Hdiv = FunctionSpace(mesh, "BDM", order + 1)
    P1 = FunctionSpace(mesh, "CG", order)

    if s_type == "fenics":
        sigma = TrialFunction(Hdiv)
        tau = TestFunction(Hdiv)
        u = TrialFunction(H1)
        v = TestFunction(H1)
        p = TrialFunction(P1)
        q = TestFunction(P1)
        p_ = TrialFunction(P1)
        q_ = TestFunction(P1)
    elif s_type == "firedrake":
        W_stokes = H1 * P1
        W_laplacian = Hdiv * P1

        (sigma, p_) = TrialFunctions(W_laplacian)
        (tau, q_) = TestFunctions(W_laplacian)
        (u, p) = TrialFunctions(W_stokes)
        (v, q) = TestFunctions(W_stokes)

    l = Re * inner(p, q) * dx
    a = (1. / Re) * inner(grad(u), grad(v)) * dx
    b = - q * div(u) * dx

    l_ = Re * inner(p_, q_) * dx
    c = div(sigma) * q_ * dx
    p = (1. / Re) * (inner(sigma, tau) + inner(div(sigma), div(tau))) * dx
    if s_type == "firedrake":
        stokes_lin = a + b + l
        laplacian_lin = p + c + l_
        bcs = [DirichletBC(W_stokes.sub(0), Constant((1, 0)), (4,)),
               DirichletBC(W_stokes.sub(0), Constant((0, 0)), [1, 2, 3])]
        bcs_primal = [DirichletBC(H1, Constant((1, 0)), (4,)),
                      DirichletBC(H1, Constant((0, 0)), [1, 2, 3])]

    e_min_primal = np.inf
    e_max_primal = -np.inf
    e_min_dual = np.inf
    e_max_dual = -np.inf
    if s_type == "fenics":
        for cell in cells(mesh):
            C = assemble_local(c, cell)
            P = assemble_local(p, cell)
            L = assemble_local(l_, cell)

            S = np.matmul(C, np.linalg.solve(P, C.T))
            e, _ = np.linalg.eig(np.linalg.solve(S, L))
            e_min_dual = min(e) if min(e) < e_min_dual else e_min_dual
            e_max_dual = max(e) if max(e) > e_max_dual else e_max_dual

            A = assemble_local(a, cell)
            B = assemble_local(b, cell)
            L = assemble_local(l, cell)

            S = A + np.matmul(B.T, np.linalg.solve(L, B))
            e, _ = np.linalg.eig(np.linalg.solve(S, A))
            e = np.real(e)
            e_min_primal = min(e) if min(e) < e_min_primal else e_min_primal
            e_max_primal = max(e) if max(e) > e_max_primal else e_max_primal
    elif s_type == "firedrake":
        AA = Tensor(laplacian_lin)
        A = AA.blocks
        dual_lin = A[1, 0] * A[0, 0].inv * A[1, 0].T
        dual = allocate_matrix(dual_lin, mat_type="aij")
        _assemble_form = create_assembly_callable(
            dual_lin, tensor=dual, mat_type="aij")
        _assemble_form()
        ai, aj, av = dual.petscmat.getValuesCSR()
        dual_ele = sp.sparse.csr_matrix((av, aj, ai))

        AA = Tensor(stokes_lin)
        A = AA.blocks
        primal_lin = A[0, 0] + A[1, 0].T * A[1, 1].inv * A[1, 0]

        primal = allocate_matrix(primal_lin, bcs=bcs_primal, mat_type="aij")
        _assemble_form = create_assembly_callable(
            primal_lin, tensor=primal, bcs=bcs_primal, mat_type="aij")
        _assemble_form()
        ai, aj, av = primal.petscmat.getValuesCSR()
        primal_ele = sp.sparse.csr_matrix((av, aj, ai))

        stokes = allocate_matrix(stokes_lin, bcs=bcs, mat_type="aij")
        _assemble_form = create_assembly_callable(
            stokes_lin, tensor=stokes, bcs=bcs, mat_type="aij")
        _assemble_form()
        ai, aj, av = stokes.petscmat.getValuesCSR()
        stokes_matrix = sp.sparse.csr_matrix((av, aj, ai))

        n = H1.dim()
        m = P1.dim()

        A = stokes_matrix[:n, :][:, :n].toarray()
        B = stokes_matrix[n:n + m, :][:, :n].toarray()
        L = stokes_matrix[n:n + m, :][:, n:n + m].toarray()

        S_primal = A + np.matmul(B.T, np.linalg.solve(L, B))
        e, _ = np.linalg.eig(np.linalg.solve(primal_ele.toarray(), S_primal))
        e_min_primal = np.real(min(e))
        e_max_primal = np.real(max(e))

        S_dual = np.matmul(B, np.linalg.solve(A, B.T))
        e, _ = np.linalg.eig(np.linalg.solve(dual_ele.toarray(), S_dual))
        sorted_e = np.real(np.sort(e))
        e_min_dual = sorted_e[1]
        e_max_dual = sorted_e[-1]

    dual_min.append(e_min_dual)
    dual_max.append(e_max_dual)
    primal_min.append(e_min_primal)
    primal_max.append(e_max_primal)
    cell_num.append(mesh.num_cells())

data = {"# cells": cell_num,
        "dual_min": dual_min,
        "dual_max": dual_max,
        "primal_min": primal_min,
        "primal_max": primal_max}

table = pd.DataFrame.from_dict(data)
print(table.to_latex())
