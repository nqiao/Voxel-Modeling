import numpy as np 
from numba import cuda
import matplotlib.pyplot as plt
import math
from time import time

TPBX = 16
TPBY = 16

# Problem 2
def f2D(x, y):
    return math.sin(np.pi * x) * math.sinh(np.pi * y) / math.sinh(np.pi)


def fArray(x, y):
    nx = x.shape[0]
    ny = y.shape[0]
    f = np.empty((nx, ny), dtype = np.float32)
    for i in range(nx):
        for j in range(ny):
            f[i, j] = f2D(x[i], y[i])
    return f



@cuda.jit(device=True)
def f2D_cuda(x, y):
    return math.sin(np.pi * x) * math.sinh(np.pi * y) / math.sinh(np.pi)


@cuda.jit('void(f4[:], f4[:], f4[:,:])')
def fKernel2D(d_x, d_y, d_f):
    i, j = cuda.grid(2)
    nx, ny = d_f.shape
    if i < nx and j < ny:
        d_f[i, j] = f2D_cuda(d_x[i], d_y[i])


def fArray2D(x, y):
    nx = x.shape[0]
    ny = y.shape[0]
    d_x = cuda.to_device(x)
    d_y = cuda.to_device(y)
    d_f = cuda.device_array((nx, ny), dtype = np.float32)
    gridDims = ((nx + TPBX - 1) // TPBX, (ny + TPBY - 1) // TPBY)
    blockDims = (TPBX, TPBY)
    fKernel2D[gridDims, blockDims](d_x, d_y, d_f)
    return d_f.copy_to_host()


# Problem 5
@cuda.reduce
def sum_reduce(x, y):
    return x + y


@cuda.jit
def norm_kernel(d_f, p):
    i, j = cuda.grid(2)
    nx, ny = d_f.shape
    if i < nx and j < ny:
        d_f[i][j] = d_f[i][j] ** p


def lp_norm(p):
    N = 256
    x = np.linspace(0, 1, 256)
    y = np.linspace(0, 1, 256)
    f = np.empty(256 * 256).reshape((256, 256))
    for i in range(256):
        for j in range(256):
            f[i][j] = math.sin(2 * np.pi * x[i]) * math.sin(2 * np.pi * y[j])
    d_f = cuda.to_device(f)
    gridDims = ((N + TPBX - 1) // TPBX, (N + TPBY - 1) // TPBY)
    blockDims = (TPBX, TPBY)
    norm_kernel[gridDims, blockDims](d_f, p)
    f = d_f.copy_to_host()
    print(f.shape)
    f_1 = f.flatten()
    return math.pow(sum_reduce(f_1), 1/p)


def serial_norm(p):
    N = 256
    x = np.linspace(0, 1, 256)
    y = np.linspace(0, 1, 256)
    f = np.empty(256 * 256).reshape((256, 256))
    for i in range(256):
        for j in range(256):
            f[i][j] = math.sin(2 * np.pi * x[i]) * math.sin(2 * np.pi * y[j])
    sum = 0
    for i in range(256):
        for j in range(256):
            sum += f[i][j] ** p
    return math.pow(sum, 1/p)


# Problem 6
DELTA_T = 0.01
EPOCH = 1000
@cuda.jit
def ode_kernel_c(d_x, d_v):
    i, j = cuda.grid(2)
    nx, ny = d_x.shape
    if i < nx and j < ny:
        for e in range(EPOCH):
            d_x[i] = d_x[i] + d_v[i] * DELTA_T
            d_v[i] = d_v[i] - d_x[i] * DELTA_T

@cuda.jit
def ode_kernel_d(d_x, d_v):
    i, j = cuda.grid(2)
    nx, ny = d_x.shape
    if i < nx and j < ny:
        for e in range(EPOCH):
            d_x[i] = d_x[i] + d_v[i] * DELTA_T
            d_v[i] = d_v[i] - (d_x[i] + 0.1 * d_v[i]) * DELTA_T


@cuda.jit
def ode_kernel_e(d_x, d_v):
    i, j = cuda.grid(2)
    nx, ny = d_x.shape
    if i < nx and j < ny:
        for e in range(EPOCH):
            d_x[i] = d_x[i] + d_v[i] * DELTA_T
            d_v[i] = d_v[i] + (-d_x[i] + 0.1 * (1 - d_x[i] **2 ) * d_v[i]) * DELTA_T


def ode_solver(num_grid):
    x = np.linspace(-3, 3, num_grid)
    v = np.linspace(-3, 3, num_grid)
    d_x = cuda.to_device(x)
    d_v = cuda.to_device(v)
    blocks = (num_grid + TPB - 1) // TPB
    threads = TPB
    ode_kernel_c[blocks, threads](d_x, d_v)
    x_res = d_x.copy_to_host()
    v_res = d_v.copy_to_host()
    print(x_res)
    print(v_res)







def main():
    # # Problem 2
    # x = np.random.rand(10000)
    # y = np.random.rand(10000)
    
    # start_s = time()
    # f_s = fArray(x, y)
    # end_s = time()
    # time_s = end_s - start_s
    
    # start_p = time()
    # f_p = fArray2D(x, y)
    # end_p = time()
    # time_p = end_p - start_p
    # print("serial", time_s)
    # # print(f_s)
    # print("parallel", time_p)
    # # print(f_p)
    

    # # Problem 5
    # p = 2
    # print("norm, p = ", p)
    # print(lp_norm(p))
    # print("serial norm, p = ", p)
    # print(serial_norm(p))


    # Problem 6
    ode_solver(61)

if __name__ == '__main__':
    main()















