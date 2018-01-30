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
    time_start = time()
    fKernel2D[gridDims, blockDims](d_x, d_y, d_f)
    time_end = time()
    exe_time = time_end - time_start
    return d_f.copy_to_host(), exe_time


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

def inf_norm():
    N = 256
    x = np.linspace(0, 1, 256)
    y = np.linspace(0, 1, 256)
    f = np.empty(256 * 256).reshape((256, 256))
    for i in range(256):
        for j in range(256):
            f[i][j] = math.sin(2 * np.pi * x[i]) * math.sin(2 * np.pi * y[j])
    res = -1
    for i in range(256):
        for j in range(256):
            res = max(res, f[i][j])
    return res


# Problem 6
# step size
DELTA_T = 0.001
# total steps
EPOCH = 10000
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
    # number of points selected on the circle
    N = 20
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


def plot3d(fvals, xvals, yvals, titlestring='',vars=['x','y','f'], idx=0):
    X,Y = np.meshgrid(xvals,yvals)
    if idx == 0:
        # fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax = fig.add_subplot(111,projection='3d')

        ax.plot_wireframe(X, Y, fvals, color='black')
        ax.set_title(titlestring)
        ax.set_xlabel(vars[0])
        ax.set_ylabel(vars[1])
        ax.set_zlabel(vars[2])
        plt.show()
    else:
        img_i = fig.add_subplot(8, 1, idx, projection='3d')

        img_i.plot_wireframe(X, Y, fvals, color='black')
        img_i.set_title(titlestring)
        img_i.set_xlabel(vars[0])
        img_i.set_ylabel(vars[1])
        img_i.set_zlabel(vars[2])


def main():
    # Problem 2
    print("Problem 2")
    print("See figure 1,2,3")
    print()
    time_ss = []
    time_ps = []
    for iter in range(10):
        x = np.random.rand(100 + 100 * iter)
        y = np.random.rand(100 + 100 * iter)
        
        start_s = time()
        for i in range(10):
            f_s = fArray(x, y)
        end_s = time()
        time_s = end_s - start_s
        time_ss.append(time_s)
        
        start_p = time()
        for i in range(100):
            f_p = fArray2D(x, y)
        end_p = time()
        time_p = end_p - start_p
        time_ps.append(time_p)
        # print("serial", time_s)
        # print(f_s)
        # print("parallel", time_p)
        # print(f_p)
    acc = []
    for i in range(10):
        acc.append(time_ss[i] / time_ps[i] * 10)
    array_sizes = [(100+100*i) for i in range(10)]
    plt.figure(1)
    plt.plot(array_sizes, acc)
    plt.xlabel("Array size")
    plt.ylabel("Acceleration")
    plt.savefig("figure1")

    plt.figure(2)
    plt.plot(array_sizes, np.array(time_ss) * 10)
    plt.xlabel("Array size")
    plt.ylabel("Time(ms)")
    plt.title("Serial time")
    plt.savefig("figure2")


    plt.figure(3)
    plt.plot(array_sizes, np.array(time_ps))
    plt.xlabel("Array size")
    plt.ylabel("Time(ms)")
    plt.title("Parallel time")
    plt.savefig("figure3")
    print()

    # Problem 3
    print("Problem 3")
    print("The largest square size is 32*32. In cuda, the largest number of threads in a block is 1024 (32*32=1024).")
    print("The error message is: numba.cuda.cudadrv.driver.CudaAPIError: [1] Call to cuLaunchKernel results in CUDA_ERROR_INVALID_VALUE")
    print()

    # Problem 4
    print("problem 4")
    print("a")
    # change these sizes to see difference
    TPBX = [4, 8, 16, 32, 4, 8]
    TPBY = [4, 8, 16, 32, 32, 16]
    time_4 = []
    for i in range(6):
        x = np.random.rand(100)
        y = np.random.rand(100)
        start_p_4 = time()
        for i in range(100):
            fArray2D(x, y)
        end_p_4 = time()
        time_4.append(end_p_4 - start_p_4)
    print(time_4)
    print("With a size 100 random initialized array, I tested block ratio of 4*4, 8*8, 32*32, 4*32, 8*16. The results are shown above: the execution times are similar. The aspect ratio won't affect much.")
    print()
    print("b")
    TPBX = 16
    TPBY = 16
    x = np.random.rand(1000)
    y = np.random.rand(1000)
    time_4b_start = time()
    f, ker_time = fArray2D(x, y)
    time_4b_end = time()
    print("Outside time :", time_4b_end-time_4b_start)
    print("Kernel time: ", ker_time*32)
    print("With a size 100 random initialized array, I tested the exection time of wrapping around fArray2D() and around the kernel function.")
    print("They are very similar. Wrapping around the kernel is a little faster, due to data transfer operation.")
    print()



    

    # Problem 5
    print("Problem 5")
    print("Input plot see figure5")
    for k in range(1):
        N = 256
        x = np.linspace(0, 1, 256)
        y = np.linspace(0, 1, 256)
        f = np.empty(256 * 256).reshape((256, 256))
        for i in range(256):
            for j in range(256):
                f[i][j] = math.sin(2 * np.pi * x[i]) * math.sin(2 * np.pi * y[j])
        plt.figure()
        plt.contourf(x,y,f)
        plt.savefig("figure5")
    p = [2, 4, 6]
    for i in range(3):
        print("L" , p[i], " norm = ", lp_norm(p[i]))
    print("Inf norm = ", inf_norm())
    print("Yes, the norm are approching L inf norm when p increases.")
    print()

    # Problem 6
    print("Problem 6")
    print("a")
    print("It's impossible to parallel the computation over a grid of time intervals, due to the current step is depends on the previous one, the data in different grids couldn't communicate. ")
    print()
    print("b")
    print("Yes, there is a way to direct parallel over a grid of initial conditions. Send one pair of initial condition (x and v) to a thread. In the kernel function, do iteration with the given relation between x and v. Since the computation only depends on its own previous state, we can parallel over different ICs. ")
    print()
    print("c")
    print("See figure6_c")
    print("Distance to the origin: ")
    state_c = ode_solver(ode_kernel_c)
    for item in state_c:
        print(pow(item[0]**2+item[1]**2, 1/2))
    plt.figure(figsize=(8,8))
    plt.plot(state_c[:,0], state_c[:,1])
    plt.savefig("figure6_c")
    
    print()
    print("d")
    print("See figure6_d")
    print("Distance to the origin: ")
    state_d = ode_solver(ode_kernel_d)
    for item in state_d:
        print(pow(item[0]**2+item[1]**2, 1/2))
    plt.figure(figsize=(8,8))
    plt.plot(state_d[:,0], state_d[:,1])
    plt.savefig("figure6_d")
    
    print()
    print("e")
    print("See figure6_e")
    print("Distance to the origin: ")
    state_e = ode_solver(ode_kernel_e)
    for item in state_e:
        print(pow(item[0]**2+item[1]**2, 1/2))
    plt.figure(figsize=(8,8))
    plt.plot(state_e[:,0], state_e[:,1])
    plt.savefig("figure6_e")
    
    print()
    print("Disscussion:")
    print("I select 20 points on the circle (r=3) in the phase diagram as ICs. Step size 0.001, totally 10000 steps. The distance between the final state and the origin")
    print("The distance between the final state and the origin: c) a sightly larger than 3. This value should keep unchaged in this case; d), e) due to damping, the distance decrease. If it runs for more steps, it would go to 0.")
    print("Only in figure_6e, points from different ICs have different distance from the origin point. Because it follows a nonliner ODE.")




if __name__ == '__main__':
    main()















