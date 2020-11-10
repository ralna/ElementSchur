import numpy as np
import pandas as pd
from scipy import linalg

from dolfin import *

N = [2, 4, 8, 16, 32]
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
    Hdiv = FunctionSpace(mesh, "BDM", order + 1)
    H1 = FunctionSpace(mesh, "CG", order)

    sigma = TrialFunction(Hdiv)
    tau = TestFunction(Hdiv)
    u = TrialFunction(Hcurl)
    v = TestFunction(Hcurl)
    p = TrialFunction(H1)
    q = TestFunction(H1)
    eps = 1e-6

    # Maxwell formulation
    l = inner(grad(p), grad(q)) * dx
    a = (1. / Re) * inner(curl(u), curl(v)) * dx
    b = inner(u, grad(q)) * dx
    x = a + inner(u, v) * dx
    l_eps = l + eps * inner(p, q) * dx
    c = (curl(u) * tau[0] + curl(u) * tau[1]) * dx
    p = Re * (dot(div(sigma), div(tau)) + dot(sigma, tau)) * dx
    a_div = inner(u, v) * dx

    e_min_primal = np.inf
    e_max_primal = -np.inf
    e_min_primal_eps = np.inf
    e_max_primal_eps = -np.inf
    e_min_dual = np.inf
    e_max_dual = -np.inf
    for cell in cells(mesh):
        A = assemble_local(a, cell)
        B = assemble_local(b, cell)
        L = assemble_local(l, cell)
        X = assemble_local(x, cell)
        L_eps = assemble_local(l_eps, cell)
        P = assemble_local(p, cell)
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

        S = A_div + np.matmul(C.T, np.linalg.solve(P, C))
        e, _ = linalg.eig(S, X)
        e = np.real(e)
        e_min_primal = min(e) if min(e) < e_min_primal else e_min_primal
        e_max_primal = max(e) if max(e) > e_max_primal else e_max_primal

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
