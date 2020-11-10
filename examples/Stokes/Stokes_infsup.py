import numpy as np
import pandas as pd
from dolfin import *
from scipy import linalg

N = [2, 4, 8, 16]
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

    sigma = TrialFunction(Hdiv)
    tau = TestFunction(Hdiv)
    u = TrialFunction(H1)
    v = TestFunction(H1)
    p = TrialFunction(P1)
    q = TestFunction(P1)
    eps = 1e-6
    l = Re * inner(p, q) * dx
    a = (1. / Re) * inner(grad(u), grad(v)) * dx
    a_eps = (1. / Re) * (inner(grad(u), grad(v)) + eps * inner(u, v)) * dx
    b = - q * div(u) * dx

    c = div(sigma) * q * dx
    p = (1. / Re) * (inner(sigma, tau) + inner(div(sigma), div(tau))) * dx

    e_min_primal = np.inf
    e_max_primal = -np.inf
    e_min_dual = np.inf
    e_max_dual = -np.inf
    e_min_dual_eps = np.inf
    e_max_dual_eps = -np.inf
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
        e = np.real(e)
        e_min_primal = min(e) if min(e) < e_min_primal else e_min_primal
        e_max_primal = max(e) if max(e) > e_max_primal else e_max_primal

        A_eps = assemble_local(a_eps, cell)
        S = np.matmul(B, np.linalg.solve(A_eps, B.T))
        e, _ = linalg.eig(S, L)
        e = np.real(e)
        e_min_dual_eps = min(e) if min(e) < e_min_dual_eps else e_min_dual_eps
        e_max_dual_eps = max(e) if max(e) > e_max_dual_eps else e_max_dual_eps

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
