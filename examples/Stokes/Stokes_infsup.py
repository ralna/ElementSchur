import numpy as np
import pandas as pd
from scipy import linalg
import scipy as sp
import matplotlib.pylab as plt
from scipy import sparse
try:
    from firedrake import *
    from firedrake.assemble import allocate_matrix, \
        create_assembly_callable
except ImportError:
    import_type = "fenics"
try:
    from dolfin import *
except ImportError:
    import_type = "firedrake"


def assemble_firedrake(dual_lin, bcs=[]):
    matrix = allocate_matrix(dual_lin, bcs=bcs, mat_type="aij")
    _assemble_form = create_assembly_callable(
        dual_lin, tensor=matrix, bcs=bcs, mat_type="aij")
    _assemble_form()
    ai, aj, av = matrix.petscmat.getValuesCSR()
    matrix_scipy = sp.sparse.csr_matrix((av, aj, ai))
    return matrix_scipy


N = [4]
Re = 100.0
dual_min = []
dual_max = []
dual_eps_min = []
dual_eps_max = []
primal_min = []
primal_max = []
cell_num = []
for n in N:
    mesh = UnitSquareMesh(n, n)

    order = 1
    H1 = VectorFunctionSpace(mesh, "CG", order + 1)
    Hdiv = FunctionSpace(mesh, "BDM", order + 1)
    P1 = FunctionSpace(mesh, "CG", order)

    if import_type == "fenics":
        sigma = TrialFunction(Hdiv)
        tau = TestFunction(Hdiv)
        u = TrialFunction(H1)
        v = TestFunction(H1)
        p = TrialFunction(P1)
        q = TestFunction(P1)
        p_ = TrialFunction(P1)
        q_ = TestFunction(P1)
    elif import_type == "firedrake":
        W_stokes = H1 * P1
        W_laplacian = Hdiv * P1

        (sigma, p_) = TrialFunctions(W_laplacian)
        (tau, q_) = TestFunctions(W_laplacian)
        (u, p) = TrialFunctions(W_stokes)
        (v, q) = TestFunctions(W_stokes)

    eps = 1e-6
    l = Re * inner(p, q) * dx
    a = (1. / Re) * inner(grad(u), grad(v)) * dx
    a_eps = ((1. / Re) * inner(grad(u), grad(v)) + eps * inner(u, v)) * dx
    b = - q * div(u) * dx

    l_ = Re * inner(p_, q_) * dx
    c = div(sigma) * q_ * dx
    p = (1. / Re) * (inner(sigma, tau) + inner(div(sigma), div(tau))) * dx
    if import_type == "firedrake":
        stokes_lin = a + b + l
        stokes_eps_lin = a_eps + b + l
        laplacian_lin = p + c + l_
        bcs = \
            [DirichletBC(W_stokes.sub(0), Constant((0, 0)), [1, 2, 3, 4])]
        bcs_primal = \
            [DirichletBC(H1, Constant((0, 0)), [1, 2, 3, 4])]

    e_min_primal = np.inf
    e_max_primal = -np.inf
    e_min_dual = np.inf
    e_max_dual = -np.inf
    e_min_dual_eps = np.inf
    e_max_dual_eps = -np.inf
    if import_type == 'fenics':
        for cell in cells(mesh):
            C = assemble_local(c, cell)
            P = assemble_local(p, cell)
            L = assemble_local(l, cell)

            S = np.matmul(C, np.linalg.solve(P, C.T))
            e, _ = linalg.eig(S, L)
            e = np.real(e)
            e_min_dual = min(e) if min(e) < e_min_dual else e_min_dual
            e_max_dual = max(e) if max(e) > e_max_dual else e_max_dual

            A = assemble_local(a, cell)
            B = assemble_local(b, cell)

            S = A + np.matmul(B.T, np.linalg.solve(L, B))
            e, _ = linalg.eig(S, A)
            e = np.real(e)
            e_min_primal = min(e) if min(e) < e_min_primal else e_min_primal
            e_max_primal = max(e) if max(e) > e_max_primal else e_max_primal

            A_eps = assemble_local(a_eps, cell)
            S = np.matmul(B, np.linalg.solve(A_eps, B.T))
            e, _ = linalg.eig(S, L)
            e = np.real(e)
            e_min_dual_eps = min(e) if min(
                e) < e_min_dual_eps else e_min_dual_eps
            e_max_dual_eps = max(e) if max(
                e) > e_max_dual_eps else e_max_dual_eps
    if import_type == 'firedrake':
        A_laplacian = Tensor(laplacian_lin)
        A = A_laplacian.blocks
        dual_lin = A[1, 0] * A[0, 0].inv * A[1, 0].T
        dual_ele = assemble_firedrake(dual_lin)

        A_stokes_eps = Tensor(stokes_eps_lin)
        A = A_stokes_eps.blocks
        dual_lin = A[1, 0] * A[0, 0].inv * A[1, 0].T
        dual_ele_eps = assemble_firedrake(dual_lin)

        A_stokes = Tensor(stokes_eps_lin)
        A = A_stokes.blocks
        primal_lin = A[0, 0] + A[1, 0].T * A[1, 1].inv * A[1, 0]
        primal_ele = assemble_firedrake(primal_lin, bcs=bcs_primal)

        stokes = assemble_firedrake(stokes_lin, bcs=bcs)
        n = H1.dim()
        m = P1.dim()
        print(m + n)
        A = stokes[:n, :][:, :n].toarray()
        B = stokes[n:n + m, :][:, :n].toarray()
        L = stokes[n:n + m, :][:, n:n + m].toarray()
        S_primal = A + np.matmul(B.T, np.linalg.solve(L, B))
        S_dual = np.matmul(B, np.linalg.solve(A, B.T))

        P_dual = sparse.block_diag([A, dual_ele]).toarray()
        P_dual_eps = sparse.block_diag([A, dual_ele_eps]).toarray()
        P_primal = sparse.block_diag([primal_ele, L]).toarray()
        K = sparse.bmat([[A, B.T], [B, None]]).toarray()

        fig = plt.figure()
        e, _ = linalg.eig(K, P_dual)
        e = np.sort(np.real(e))
        plt.plot(e, "o", label="$\\mathcal{A}x = \\lambda \\mathcal{P}_1x$")
        e, _ = linalg.eig(K, P_dual_eps)
        e = np.sort(np.real(e))
        plt.plot(e, "x", label="$\\mathcal{A}x = \\lambda \\mathcal{P}_2x$")
        e, _ = linalg.eig(K, P_primal)
        e = np.sort(np.real(e))
        plt.plot(e, "+", label="$\\mathcal{A}x = \\lambda \\mathcal{P}_3x$")
        plt.legend()
        fig.savefig("stokes.png")
    dual_min.append(e_min_dual)
    dual_max.append(e_max_dual)
    dual_eps_min.append(e_min_dual_eps)
    dual_eps_max.append(e_max_dual_eps)
    primal_min.append(e_min_primal)
    primal_max.append(e_max_primal)
    cell_num.append(mesh.num_cells())

data = {"# cells": cell_num,
        "dual_min": dual_min,
        "dual_max": dual_max,
        "dual_eps_min": dual_eps_min,
        "dual_eps_max": dual_eps_max,
        "primal_min": primal_min,
        "primal_max": primal_max}

table = pd.DataFrame.from_dict(data)
print(table.to_latex())
