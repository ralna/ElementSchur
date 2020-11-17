import numpy as np
import pandas as pd
from scipy import linalg
import scipy as sp
from scipy import sparse
import matplotlib.pylab as plt
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


N = [6]
Re = 100.0
dual_min = []
dual_max = []
primal_min = []
primal_max = []
primal_eps_min = []
primal_eps_max = []
cell_num = []
for n in N:
    mesh = UnitSquareMesh(n, n)

    order = 1
    Hcurl = FunctionSpace(mesh, "N1curl", order)
    Hdiv = FunctionSpace(mesh, "BDM", order)
    H1 = FunctionSpace(mesh, "CG", order)
    if import_type == "fenics":
        sigma = TrialFunction(Hdiv)
        tau = TestFunction(Hdiv)
        u = TrialFunction(Hcurl)
        v = TestFunction(Hcurl)
        u_ = TrialFunction(Hcurl)
        v_ = TestFunction(Hcurl)
        p = TrialFunction(H1)
        q = TestFunction(H1)
    elif import_type == "firedrake":
        W_maxwell = Hcurl * H1
        W_hodge = Hcurl * Hdiv

        (u_, sigma) = TrialFunctions(W_hodge)
        (v_, tau) = TestFunctions(W_hodge)
        (u, p) = TrialFunctions(W_maxwell)
        (v, q) = TestFunctions(W_maxwell)
    eps = 1e-6

    # Maxwell formulation
    l = inner(grad(p), grad(q)) * dx
    a = (1. / Re) * inner(curl(u), curl(v)) * dx
    b = inner(u, grad(q)) * dx
    x = a + inner(u, v) * dx
    l_eps = l + eps * inner(p, q) * dx
    c = (curl(u_) * tau[0] + curl(u_) * tau[1]) * dx
    d = Re * (dot(div(sigma), div(tau)) + dot(sigma, tau)) * dx
    a_div = inner(u_, v_) * dx
    if import_type == "firedrake":
        maxwell_eps_lin = a + b + l_eps
        maxwell_lin = a + b + l
        maxwell_lin_norm = x + b
        hodge_lin = a_div + d + c
        bcs = \
            [DirichletBC(W_maxwell.sub(0), Constant((0, 0)), [1, 2, 3, 4]),
             DirichletBC(W_maxwell.sub(1), Constant((0)), [1, 2, 3, 4])]
        bcs_primal = \
            [DirichletBC(Hcurl, Constant((0, 0)), [1, 2, 3, 4])]
        bcs_dual = \
            [DirichletBC(H1, Constant(0), [1, 2, 3, 4])]
    e_min_primal = np.inf
    e_max_primal = -np.inf
    e_min_primal_eps = np.inf
    e_max_primal_eps = -np.inf
    e_min_dual = np.inf
    e_max_dual = -np.inf
    if import_type == 'fenics':
        for cell in cells(mesh):
            A = assemble_local(a, cell)
            B = assemble_local(b, cell)
            L = assemble_local(l, cell)
            X = assemble_local(x, cell)
            L_eps = assemble_local(l_eps, cell)
            D = assemble_local(d, cell)
            C = assemble_local(c, cell)
            A_div = assemble_local(a_div, cell)

            S = np.matmul(B, np.linalg.solve(X, B.T))
            e, _ = linalg.eig(S, L)
            e = np.real(e)
            inf_array = np.isinf(e)
            not_inf_array = ~inf_array
            e = e[not_inf_array]
            e_min_dual = min(e) if min(e) < e_min_dual else e_min_dual
            e_max_dual = max(e) if max(e) > e_max_dual else e_max_dual

            S = A + np.matmul(B.T, np.linalg.solve(L_eps, B))
            e, _ = linalg.eig(S, X)
            e = np.real(e)
            e_min_primal_eps = min(e) if min(e) < e_min_primal_eps \
                else e_min_primal_eps
            e_max_primal_eps = max(e) if max(e) > e_max_primal_eps \
                else e_max_primal_eps

            S = A_div + np.matmul(C.T, np.linalg.solve(D, C))
            e, _ = linalg.eig(S, X)
            e = np.real(e)
            e_min_primal = min(e) if min(e) < e_min_primal \
                else e_min_primal
            e_max_primal = max(e) if max(e) > e_max_primal \
                else e_max_primal
    if import_type == 'firedrake':
        A_maxwell = Tensor(maxwell_lin_norm)
        A = A_maxwell.blocks
        dual_lin = A[1, 0] * A[0, 0].inv * A[1, 0].T
        dual_ele = assemble_firedrake(dual_lin, bcs=bcs_dual)

        A_maxwell_eps = Tensor(maxwell_eps_lin)
        A = A_maxwell_eps.blocks
        primal_lin = A[0, 0] + A[1, 0].T * A[1, 1].inv * A[1, 0]
        primal_ele_eps = assemble_firedrake(primal_lin, bcs=bcs_primal)

        A_hodge = Tensor(hodge_lin)
        A = A_hodge.blocks
        primal_lin = A[0, 0] + A[1, 0].T * A[1, 1].inv * A[1, 0]
        primal_ele = assemble_firedrake(primal_lin, bcs=bcs_primal)

        maxwell = assemble_firedrake(maxwell_lin, bcs=bcs)
        maxwell_norm = assemble_firedrake(maxwell_lin_norm, bcs=bcs)
        n = Hcurl.dim()
        m = H1.dim()
        print(n + m)
        A = maxwell[:n, :][:, :n].toarray()
        B = maxwell[n:n + m, :][:, :n].toarray()
        L = maxwell[n:n + m, :][:, n:n + m].toarray()
        dual_BC = maxwell_norm[n:n + m, :][:, n:n + m].toarray()
        X = maxwell_norm[:n, :][:, :n].toarray()
        S_primal = A + np.matmul(B.T, np.linalg.solve(L, B))

        P_dual = sparse.block_diag([X, dual_ele]).toarray()
        P_primal = sparse.block_diag([primal_ele, L]).toarray()
        P_primal_eps = sparse.block_diag([primal_ele_eps, L]).toarray()
        K = sparse.bmat([[A, B.T], [B, dual_BC]]).toarray()

        fig = plt.figure()
        e, _ = linalg.eig(K, P_dual)
        e = np.sort(np.real(e))
        plt.plot(e, "o", label="$\\mathcal{A}x = \\lambda \\mathcal{P}_1x$")
        e, _ = linalg.eig(K, P_primal_eps)
        e = np.sort(np.real(e))
        plt.plot(e, "x", label="$\\mathcal{A}x = \\lambda \\mathcal{P}_2x$")
        e, _ = linalg.eig(K, P_primal)
        e = np.sort(np.real(e))
        plt.plot(e, "+", label="$\\mathcal{A}x = \\lambda \\mathcal{P}_3x$")
        plt.legend()
        fig.savefig('maxwell.png')
    dual_min.append(e_min_dual)
    dual_max.append(e_max_dual)
    primal_min.append(e_min_primal)
    primal_max.append(e_max_primal)
    primal_eps_min.append(e_min_primal_eps)
    primal_eps_max.append(e_max_primal_eps)
    cell_num.append(mesh.num_cells())

data = {"# cells": cell_num,
        "dual_min": dual_min,
        "dual_max": dual_max,
        "primal_min": primal_min,
        "primal_max": primal_max,
        "primal_eps_min": primal_eps_min,
        "primal_eps_max": primal_eps_max}
table = pd.DataFrame.from_dict(data)
print(table.to_latex())
