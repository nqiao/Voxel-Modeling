import numpy as np 
from numba import cuda
import matplotlib.pyplot as plt
import math
from time import time

TPBX = 16
TPBY = 16
TPB = 32

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
def ode_kernel_c(d_f):
    i = cuda.grid(1)
    n = d_f.shape[0]
    if i < n:
        for e in range(EPOCH):
            x = d_f[i][0]
            v = d_f[i][1]
            d_f[i][0] = x + v * DELTA_T
            d_f[i][1] = v - x * DELTA_T

@cuda.jit
def ode_kernel_d(d_f):
    i = cuda.grid(1)
    n = d_f.shape[0]
    if i < n:
        for e in range(EPOCH):
            x = d_f[i][0]
            v = d_f[i][1]
            d_f[i][0] = x + v * DELTA_T
            d_f[i][1] = v - (x + 0.1 * v) * DELTA_T


@cuda.jit
def ode_kernel_e(d_f):
    i = cuda.grid(1)
    n = d_f.shape[0]
    if i < n:
        for e in range(EPOCH):
            x = d_f[i][0]
            v = d_f[i][1]
            d_f[i][0] = x + v * DELTA_T
            d_f[i][1] = v + (-x + 0.1 * (1 - x **2 ) * v) * DELTA_T


def ode_solver(func):
    function = func
    N = 100
    scale = np.linspace(0, 1, N)
    f = np.empty(N * 2).reshape((N, 2))
    for i in range(N):
        f[i][0] = 3 * math.cos(2*np.pi*scale[i])
        f[i][1] = 3 * math.sin(2*np.pi*scale[i])
    d_f = cuda.to_device(f)
    gridDims = (N + TPB - 1) // TPB
    blockDims = TPB
    # Switch kernel here: c, d, e
    function[gridDims, blockDims](d_f)
    f = d_f.copy_to_host()
    return f


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

    # Problem 3
    print("Problem 3")
    print("The largest square size is 32*32. In cuda, the largest number of threads in a block is 1024 (32*32=1024).")
    print("The error message is: numba.cuda.cudadrv.driver.CudaAPIError: [1] Call to cuLaunchKernel results in CUDA_ERROR_INVALID_VALUE")
    

    # # Problem 5
    # p = 2
    # print("norm, p = ", p)
    # print(lp_norm(p))
    # print("serial norm, p = ", p)
    # print(serial_norm(p))


    # Problem 6
    print("Problem 6")
    print("a")
    print("It's impossible to parallel the computation over a grid of time intervals, due to the current step is depends on the previous one, the data in different grids couldn't communicate. ")
    print("b")
    print("Yes, there is a way to direct parallel over a grid of initial conditions. Send one pair of initial condition (x and v) to a thread. In the kernel function, do iteration with the given relation between x and v. Since the computation only depends on its own previous state, we can parallel over different ICs. ")
    print("c")
    print("Distance to the origin: ")
    state_c = ode_solver(ode_kernel_e)
    print(state_c.reshape(-1,2))
    plt.figure(figsize=(8,8))
    plt.plot(state_c[:,0], state_c[:,1])
    plt.show()
    # for item in state_c:
    #     print(pow(item[0]**2+item[1]**2, 1/2))

if __name__ == '__main__':
    main()















