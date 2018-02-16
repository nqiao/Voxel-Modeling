import numpy as np
import matplotlib.pyplot as plt
from plotUtils import *
import copy
from time import time
import math
from numba import cuda, float32, jit, int32



# Problem 1
# 1.a
def analysis(ori_arr):
    """Analyze the input array, return the first array and the second array."""
    even_res = []
    odd_res = []
    for i in range(len(ori_arr)):
        if i % 2 == 0:
            even_res.append((ori_arr[i+1] + ori_arr[i]) / 2)
        else:
            odd_res.append((ori_arr[i] - ori_arr[i-1]) / 2)
    return np.array(even_res), np.array(odd_res)

def plot_analysis(ori_arr, first_arr, second_arr, fig_name="no_name"):
    n_arr = len(ori_arr)
    plt.figure()
    plt.plot(ori_arr, 'o', label='original array')
    plt.plot([i for i in range(0,n_arr-1,2)], first_arr, 'o', label='first array')
    plt.plot([i for i in range(1,n_arr,2)], second_arr, 'o', label='second array')
    plt.legend()
    plt.savefig(fig_name + ".jpg")

def synthesis(first_arr, second_arr):
    """Do synthesis based on the two given arrays, return the reconstructed array."""
    syn_arr = np.stack((first_arr, second_arr))
    syn_arr = syn_arr.T
    syn_arr = syn_arr.flatten()
    n_arr = len(syn_arr)
    syn_res = np.empty(n_arr)
    for i in range(n_arr):
        if i % 2 == 0:
            syn_res[i] = syn_arr[i] - syn_arr[i+1]
        else:
            syn_res[i] = syn_arr[i-1] + syn_arr[i]    
    return syn_res


# Function to solve problem 2.a
def serial_laplace(u, epoch=10):
    """A serial Laplace function solver"""
    n_arr = len(u)
    for e in range(epoch):
        for i in range(1, n_arr-1):
            u[i] = (u[i-1] + u[i+1]) / 2
    return u

# Execution function to 2.a
def excute_2a(epoch=100):
    print("----- 2.a -----")
    init_arr = gen_ic(16)
    res = serial_laplace(init_arr, epoch)
    lin = np.linspace(0,1,16)
    diff = res - lin
    print("Epochs: ", epoch)
    print("Serial Laplace result: ")
    print(res)
    print("Difference with the correct result")
    print(diff)
    print()

def gen_ic(n):
    """Generate the initial array for problem 2."""
    init_arr = np.zeros(n)
    init_arr[-1] = 1
    return init_arr


# Kernel function to global memory Laplace solver
@cuda.jit
def laplace_kernel(d_u, epoch):
    i = cuda.grid(1)
    n = d_u.shape[0]
    for e in range(epoch):
        if i >= n - 1 or i == 0:
            return
        d_u[i] = (d_u[i-1] + d_u[i+1]) / 2

# Wrapper function to global Laplace solver
def nu_laplace(u, epoch=10):
    TPB = 8
    n = u.shape[0]
    d_u = cuda.to_device(u)
    blocks = (n + TPB - 1) // TPB
    threads = TPB
    laplace_kernel[blocks, threads](d_u, epoch)
    return d_u.copy_to_host()

# Execution function to 2.b
def excute_2b(epoch):
    print("----- 2.b -----")
    init_arr = gen_ic(16)
    res = nu_laplace(init_arr, epoch)
    lin = np.linspace(0,1,16)
    diff = res - lin
    print("Epochs: ", epoch)
    print("Serial Laplace result: ")
    print(res)
    print("Difference with the correct result")
    print(diff)
    print()


# 2.c/d/e
# TPB = 16
# NSHARED = 18

# Kernel function to 2.c/d, RAD=1
@cuda.jit
def share_laplace_kernel_1(d_u, stencil, epoch):
    TPB = 16
    NSHARED = 18
    i = cuda.grid(1)
    n = d_u.shape[0]
    sh_u = cuda.shared.array(NSHARED, dtype=float32)
    
    if i >= n or i < 0:
        return
    
    radius = len(stencil) // 2
    tIdx = cuda.threadIdx.x
    shIdx = tIdx + radius
    sh_u[shIdx] = d_u[i]
    
    if tIdx < radius:
        sh_u[tIdx] = d_u[i - radius]
    elif tIdx > cuda.blockDim.x - 1 - radius:
        sh_u[tIdx + 2 * radius] = d_u[i + radius]
      
    cuda.syncthreads()
    
    if radius <= i <= n - 1 - radius:
        d_u[i] = 0
        for j in range(len(stencil)):       
            d_u[i] += (sh_u[shIdx + j - radius] * stencil[j])


# Kernel function to 2.e, RAD=2
@cuda.jit
def share_laplace_kernel_2(d_u, stencil, epoch):
    """Kernel function for problem 2.e, RAD=2."""
    TPB = 16
    NSHARED = 20
    i = cuda.grid(1)
    n = d_u.shape[0]
    sh_u = cuda.shared.array(NSHARED, dtype=float32)
    
    if i >= n or i < 0:
        return
    
    radius = len(stencil) // 2
    tIdx = cuda.threadIdx.x
    shIdx = tIdx + radius
    sh_u[shIdx] = d_u[i]
    
    if tIdx < radius:
        sh_u[tIdx] = d_u[i - radius]
    elif tIdx > cuda.blockDim.x - 1 - radius:
        sh_u[tIdx + 2 * radius] = d_u[i + radius]

        
    cuda.syncthreads()

    if i == 1:
        d_u[i] = 0
    if i == n - 2:
        d_u[i] = 1
    
    w = 0.5
    if radius <= i <= n - 1 - radius:
        temp = 0
        for j in range(len(stencil)):       
            temp += (sh_u[shIdx + j - radius] * stencil[j])
        d_u[i] = w * temp + (1 - w) * d_u[i]
            
# Wrapper function of shared memory Laplace solver
def share_laplace(u, order=1, epoch=100, tol=1e-5):
    TPB = 16
    n = u.shape[0]
    
    if order == 1:
        stencil = np.array([1., 0., 1.], dtype=np.float32) / 2
        share_laplace_kernel = share_laplace_kernel_1
    elif order == 2:
        stencil = np.array([-1./12, 4./3, 0., 4./3, -1./12], dtype=np.float32) * 2 / 5
        share_laplace_kernel = share_laplace_kernel_2
        
    d_u = cuda.to_device(u)
    
    blocks = (n + TPB - 1) // TPB
    threads = TPB
    
    for e in range(epoch):
        pre_u = d_u.copy_to_host()
        share_laplace_kernel[blocks, threads](d_u, stencil, epoch)
        post_u = d_u.copy_to_host()
        diff = np.linalg.norm(post_u - pre_u)
        if diff < tol:
            print(e)
            break

    return d_u.copy_to_host()


# Execution function to problem 2.c/d/e
def excute_2cde(problem):
    if problem == 'c':
        order = 1
        epoch = 100
        tol = 1e-20
        
    elif problem == 'd':
        order = 1
        epoch = 10000
        tol = 1e-7
    elif problem == 'e':
        order = 2
        epoch = 10000
        tol = 1e-7
    s = "----- 2." + problem + " -----"    
    print(s)
    init_arr = gen_ic(16)
    print("Epochs excuted: ")
    res = share_laplace(init_arr, order, epoch, tol)
    lin = np.linspace(0,1,16)
    diff = res - lin
    print("Current tolerance: ")
    print(tol)
    print("Serial Laplace result: ")
    print(res)
    print("Difference with the correct result")
    print(diff)
    print()


# Generate u matrix for problem 3
def gen_u():
    x_scale = np.linspace(-2,2,64)
    y_scale = np.linspace(-2,2,64)
    u = np.empty(64*64).reshape(64,64)
    for i in range(64):
        for j in range(64):
            u[i,j] = min_dis(x_scale[i], y_scale[j])
    return u

 # Helper function to gen_u(). Calculate the shortest distance to three fixed points.           
def min_dis(x, y):
    dis1 = (x - 1) ** 2 + y ** 2
    dis2 = (x + 1) ** 2 + y ** 2
    dis3 = x ** 2 + (y - 1) ** 2
    min_sq = min(dis1, min(dis2, dis3))
    return math.sqrt(min_sq)

# execution function for 3.a
def excute_3a():
    x_scale = np.linspace(-2,2,64)
    y_scale = np.linspace(-2,2,64)
    u = gen_u()
    plot3d(u.T, x_scale, y_scale, titlestring='3.a_3Dplot: u',vars=['x','y','f'])
    fig = plt.figure()
    plt.contourf(u.T)
    plt.title("3.a_contoutplot: u")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('square')
    plt.show()


# Kernel function to calculate del u
@cuda.jit
def del_kernel(d_u, d_out):
    i, j = cuda.grid(2)
    nx, ny = d_u.shape
    if i < nx - 1 and j < ny - 1 and i > 0 and j > 0:
        d_out[i-1, j-1] = 0.25 * (d_u[i+1, j] - d_u[i-1, j]) ** 2 + 0.25 * (d_u[i, j+1] - d_u[i, j-1]) ** 2

# Wrapper function to calculate del u
def nu_del():
    TPB = 16
    u = gen_u()
    n = u.shape[0]
    d_u = cuda.to_device(u)
    d_out = cuda.device_array((n - 2, n - 2), dtype = np.float32)
    gridDims = ((n + TPB - 1) // TPB, (n + TPB - 1) // TPB)
    blockDims = (TPB, TPB)
    del_kernel[gridDims, blockDims](d_u, d_out)
    return d_out.copy_to_host()

# Execution function to 3.b
def excute_3b():
    x_scale = np.linspace(-2,2,64)
    y_scale = np.linspace(-2,2,64)
    del_u = nu_del()
    plot3d(del_u, x_scale[1:63], y_scale[1:63], titlestring='3.b_3Dplot: |del_u|^2',vars=['x','y','f'])
    

# Kernel to shared memory 3c
@cuda.jit
def share_del_kernel(d_u, d_out):
    RAD = 1
    SH_N = 10
    i, j = cuda.grid(2)
    dims = d_u.shape
    if i >= dims[0] or j >= dims[1]:
        return 
    
    t_i = cuda.threadIdx.x
    t_j = cuda.threadIdx.y

    NX = cuda.blockDim.x
    NY = cuda.blockDim.y

    sh_i = t_i + RAD
    sh_j = t_j + RAD

    sh_u = cuda.shared.array(shape = (SH_N,SH_N), dtype = float32)

    #Load regular values
    sh_u[sh_i, sh_j] = d_u[i, j]
    
    #Halo edge values
    if t_i<RAD:
        sh_u[sh_i - RAD, sh_j] = d_u[i-RAD, j]
        sh_u[sh_i + NX , sh_j] = d_u[i+NX , j]

    if t_j<RAD:
        sh_u[sh_i, sh_j - RAD] = d_u[i, j - RAD]
        sh_u[sh_i, sh_j + NY ] = d_u[i, j + NY ]
        
    cuda.syncthreads()
    
    if i > 0 and j > 0 and i < dims[0] - 1 and j < dims[1] - 1:
        d_out[i-1, j-1] = 0.25 * (sh_u[sh_i+1, sh_j] - sh_u[sh_i-1, sh_j]) ** 2 + 0.25 * (sh_u[sh_i, sh_j+1] - sh_u[sh_i, sh_j-1]) ** 2
      
# Wrapper to shared memory 3c
def share_del():
    TPB = 8
    u = gen_u()
    d_u = cuda.to_device(u)
    dims = u.shape
    d_out = cuda.device_array((dims[0] - 2, dims[1] - 2), dtype = np.float32)
    
    gridSize = [(dims[0]+TPB-1)//TPB, (dims[1]+TPB-1)//TPB]
    blockSize = [TPB, TPB]
    share_del_kernel[gridSize, blockSize](d_u, d_out)
    return d_out.copy_to_host()

def measure_time():
    start_time_g = time()
    for t in range(100):
        del_u = nu_del()
    end_time_g = time()

    start_time_s = time()
    for t in range(100):
        del_u = share_del()
    end_time_s = time()

    return (end_time_g - start_time_g) / (end_time_s - start_time_s)


# Execution function to 3c
def excute_3c():
    x_scale = np.linspace(-2,2,64)
    y_scale = np.linspace(-2,2,64)
    share_del_u = share_del()
    plot3d(share_del_u, x_scale[1:63], y_scale[1:63], titlestring='3.c_3Dplot: |del_u|^2, shared memory',vars=['x','y','f'])
    global_del_u = nu_del()
    print("Difference between share memory del_u and global memory del_u: ")
    diff = share_del_u - global_del_u
    print(diff)
    print("L2 norm of differnence: ", np.linalg.norm(diff))
    print("The results are unchanged.")
    print()
    print("Acceleration = global memory time / shared memory time = ", measure_time())

# Generate the unit disk matrix
def gen_f():
    x_scale = np.linspace(-2,2,64)
    y_scale = np.linspace(-2,2,64)
    f = np.empty(64*64).reshape(64,64)
    for i in range(64):
        for j in range(64):
            f[i,j] = x_scale[i] ** 2 + y_scale[j] ** 2 - 1
    return f

# Kernel function to do upwind update
@cuda.jit
def upwind_kernel(d_u):
    i, j = cuda.grid(2)
    nx, ny = d_u.shape
    if i < nx - 1 and j < ny - 1 and i > 0 and j > 0:
        if d_u[i, j] > 0:
            t1 = min(d_u[i-1, j], d_u[i+1, j])
            t2 = min(d_u[i, j-1], d_u[i, j+1])
            h = 4 / (nx - 1)
            a = 2
            b = -2 * (t1 + t2)
            c = t1 ** 2 + t2 ** 2 - h ** 2
            delta = b ** 2 - 4 * a * c
            if delta >= 0:
                t = (-b + math.sqrt(delta)) / (2*a)
                if t > max(t1, t2):
                    d_u[i, j] = t
                elif t2 > t > t1:
                    d_u[i, j] = t1 + h
                elif t1 > t > t2:
                    d_u[i, j] = t2 + h

# Wrapper function to do upwind update
def nu_upwind(f, epoch=1):
    TPB=8
    n = f.shape[0]
    d_f = cuda.to_device(f)
    gridDims = ((n + TPB - 1) // TPB, (n + TPB - 1) // TPB)
    blockDims = (TPB, TPB)
    for e in range(epoch):
        upwind_kernel[gridDims, blockDims](d_f)
    return d_f.copy_to_host()




def main():
    # # Problem 1
    # print("##### Problem 1 #####")

    # # 1.b
    # print("----- 1.b -----")
    # test_1 = np.ones(16)
    # test_2 = np.linspace(0, 1, 16)
    # test_3 = np.linspace(0, 1, 16) ** 2
    # test_4 = np.random.rand(16)
    # print("test_1: ", test_1)
    # print()
    # print("test_2: ", test_2)
    # print()
    # print("test_3: ", test_3)
    # print()
    # print("test_4: ", test_4)
    # print()

    # # 1.c
    # print("----- 1.c -----")
    # test_1_even, test_1_odd = analysis(test_1)
    # plot_analysis(test_1, test_1_even, test_1_odd, fig_name='1.c.1')
    # print("test_1 first array: ", test_1_even)
    # print("test_1 second array: ", test_1_odd)
    # print("Plot see figure test_1")
    # print()

    # test_2_even, test_2_odd = analysis(test_2)
    # plot_analysis(test_2, test_2_even, test_2_odd, fig_name='1.c.2')
    # print("test_2 first array: ", test_2_even)
    # print("test_2 second array: ", test_2_odd)
    # print("Plot see figure test_2")
    # print()

    # test_3_even, test_3_odd = analysis(test_3)
    # plot_analysis(test_3, test_3_even, test_3_odd, fig_name='1.c.3')
    # print("test_3 first array: ", test_3_even)
    # print("test_3 second array: ", test_3_odd)
    # print("Plot see figure test_3")
    # print()

    # test_4_even, test_4_odd = analysis(test_4)
    # plot_analysis(test_4, test_4_even, test_4_odd, fig_name='1.c.4')
    # print("test_4 first array: ", test_4_even)
    # print("test_4 second array: ", test_4_odd)
    # print("Plot see figure test_4")
    # print()

    # # 1.d
    # print("----- 1.d -----")
    # syn_1 = synthesis(test_1_even, test_1_odd)
    # print("synthesis result 1: ", syn_1)
    # print("difference with the original array: ", syn_1 - test_1)
    # print()

    # syn_2 = synthesis(test_2_even, test_2_odd)
    # print("synthesis result 2: ", syn_2)
    # print("difference with the original array: ", syn_2 - test_2)
    # print()

    # syn_3 = synthesis(test_3_even, test_3_odd)
    # print("synthesis result 3: ", syn_3)
    # print("difference with the original array: ", syn_3 - test_3)
    # print()

    # syn_4 = synthesis(test_4_even, test_4_odd)
    # print("synthesis result 4: ", syn_4)
    # print("difference with the original array: ", syn_4 - test_4)
    # print()

    # # 1.e
    # print("----- 1.e -----")
    # zero_array = np.zeros(8)
    # syn_1_0 = synthesis(test_1_even, zero_array)
    # print("synthesis result 1: ", syn_1_0)
    # print("difference with the original array: ", syn_1_0 - test_1)
    # print()

    # syn_2_0 = synthesis(test_2_even, zero_array)
    # print("synthesis result 2: ", syn_2_0)
    # print("difference with the original array: ", syn_2_0 - test_2)
    # print()

    # syn_3_0 = synthesis(test_3_even, zero_array)
    # print("synthesis result 3: ", syn_3_0)
    # print("difference with the original array: ", syn_3_0 - test_3)
    # print()

    # syn_4_0 = synthesis(test_4_even, zero_array)
    # print("synthesis result 4: ", syn_4_0)
    # print("difference with the original array: ", syn_4_0 - test_4)
    # print()


    # # Problem 2
    # print("##### Problem 2 #####")
    # # epoch=100
    # excute_2a(100) 
    # # epoch=100
    # excute_2b(100)
    # excute_2cde('c')
    # excute_2cde('d')
    # excute_2cde('e')
    # print("I set the threshold to 1e-5 and 1e-7. ")
    # print("threshold        1e-5             1e-7")
    # print("RAD = 1           389              591")
    # print("RAD = 2           630              1001")
    # print("The conclusion of this experiment is it converges faster when rad=1 than rad=2.")
    # print("Notes: RAD=2, a fraction coefficient w is needed to make sure the results dont blow up.")
    # print("       Here w = 0.5.")


    # # Problem 3
    # # 3.a
    # print("##### Problem 3 #####")
    # print("----- 3.a -----")
    # print("3D plot and contourplot see pop out windows.")
    # print("Disscussion: ")
    # excute_3a()
    # print()
    # # 3.b
    # print("----- 3.b -----")
    # print("3D plot see pop out window.")
    # time_3b = excute_3b()
    # print()
    # # 3.c
    # print("----- 3.c -----")
    # print("3D plot see pop out window.")
    # excute_3c()
    # print()

    # Problem 4
    # 4.a
    print("##### Problem 4 #####")
    print("----- 4.a -----")
    print("Visualize the unit disk. See the pop out figure.")
    x_scale = np.linspace(-2,2,64)
    y_scale = np.linspace(-2,2,64)
    f_0 = gen_f()
    plot3d(f_0, x_scale, y_scale, titlestring='4.a: Unit disk',vars=['x','y','f'])
    print()

    print("----- 4.c -----")
    print("Plot the results after 10 and 20 iterations.")
    f_10 = nu_upwind(f_0, 10)
    plot3d(f_10, x_scale, y_scale, titlestring='4.c: after 10 iterations',vars=['x','y','f'])
    f_20 = nu_upwind(f_0, 20)
    plot3d(f_20, x_scale, y_scale, titlestring='4.c: after 20 iterations',vars=['x','y','f'])
    print()

    print("----- 4.d -----")
    print("Change the sign, plot after 8 and 16 iterations which is totally 28 and 36 iterations.")
    f_28 = nu_upwind(-f_20, 8)
    plot3d(f_28, x_scale, y_scale, titlestring='4.d: after 28 iterations',vars=['x','y','f'])
    f_36 = nu_upwind(-f_20, 16)
    plot3d(f_28, x_scale, y_scale, titlestring='4.d: after 36 iterations',vars=['x','y','f'])
    print()

    print("----- 4.e -----")
    print("Flip the sign again, plot the result after another 100 iterations which is 136 iterations totally.")
    print("Plot the results after 136 iterations directly from the original unit disk and compare.")
    f_136 = nu_upwind(-f_36, 100)
    plot3d(f_136, x_scale, y_scale, titlestring='4.e: after 136 iterations',vars=['x','y','f'])
    # 136 iterations directly from f_0 with no sign changed.
    f_136_dir = nu_upwind(f_0, 136)
    plot3d(f_136_dir, x_scale, y_scale, titlestring='4.e: with no sign changes, after 136 iterations',vars=['x','y','f'])
    print()

if __name__ == "__main__":
    main()