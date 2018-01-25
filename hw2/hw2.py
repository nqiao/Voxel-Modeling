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
def f2D(x, y):
    return math.sin(np.pi * x) * math.sinh(np.pi * y) / math.sinh(np.pi)


@cuda.jit('void(f4[:], f4[:], f4[:,:])')
def fKernel2D(d_x, d_y, d_f):
    i, j = cuda.grid(2)
    nx, ny = d_f.shape
    if i < nx and j < ny:
        d_f[i, j] = f2D(d_x[i], d_y[i])


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


@cuda.reduce
def sum_reduce(x, y):
    return x + y


@cuda.jit
def normKernel(d_x, d_y, d_f, p):
    i, j = cuda.grid(2)
    nx, ny = d_f.shape
    if i < nx and j < ny:
        d_f[i, j] = math.sin(d_x[i]) ** p + math.sin(d_y[j]) ** p


def lp_norm(p):
    N = 256
    scale = np.linspace(0, 2 * np.pi, 256)
    x = np.sin(scale)
    y = np.sin(scale)
    d_x = cuda.to_device(x)
    d_y = cuda.to_device(y)
    d_f = cuda.device_array((N, N), dtype = np.float32)
    gridDims = ((N + TPBX - 1) // TPBX, (N + TPBY - 1) // TPBY)
    blockDims = (TPBX, TPBY)
    normKernel[gridDims, blockDims](d_x, d_y, d_f, p)
    f = d_f.copy_to_host()
    f_1 = f.flatten()
    return math.pow(sum_reduce(f_1), 1/p)


def serial_norm(p):
    N = 256
    scale = np.linspace(0, 2 * np.pi, 256)
    x = np.sin(scale)
    y = np.sin(scale)
    sum = 0
    for i in range(256):
        for j in range(256):
            sum += x[i] ** p + y[j] ** p
    return math.pow(sum, 1/p)

def main():
    # x = np.random.rand(100)
    # y = np.random.rand(100)
    
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
    
    p = 2
    print("norm, p = ", p)
    # print(lp_norm(p))
    print("serial norm, p = ", p)
    print(serial_norm(p))

if __name__ == '__main__':
    main()














