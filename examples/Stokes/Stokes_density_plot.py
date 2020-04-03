from firedrake import *
from firedrake.petsc import PETSc
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

n = 8
mesh = UnitSquareMesh(n, n)
V = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)
Z = V * Q
Re = 10.

bcs = [DirichletBC(Z.sub(0), Constant((1, 0)), (4,)),
       DirichletBC(Z.sub(0), Constant((0, 0)), [1, 2, 3])]

u, p = TrialFunctions(Z)
v, q = TestFunctions(Z)
a = (
    (1. / Re) * inner(grad(u), grad(v)) * dx
    - p * div(v) * dx
    - q * div(u) * dx
    + Re * inner(q, p) * dx
)

a_tensor = Tensor(a)
a_blocks = a_tensor.blocks
A = assemble(a_blocks[0, 0], mat_type="aij")
B = assemble(a_blocks[1, 0], mat_type="aij")
M = assemble(a_blocks[1, 1], mat_type="aij")
A = A.M.values
B = B.M.values
M = M.M.values
S = np.matmul(B, np.linalg.solve(A, B.T))

cmap = cm.get_cmap('gray_r')
for matrix in [S, M]:
    plt.figure()
    plt.imshow(matrix, cmap)
    plt.clim(0, 0.06)
    plt.colorbar()
plt.show()
