import numpy as np #import scientific computing library
from numba import cuda
import matplotlib.pyplot as plt # import plotting library
# define values for global variables
TPB = 8 #number of threads per block


#function definitions for problem 2
def gen_array_1(n):
    """Generate an array of n values equally spaced over the
    interval [0,2*Pi] using for loop.
    """
    if n < 0:
        return None
    if n == 0:
        return np.array([])
    if n == 1:
        return np.array([0])
    interval = 2 * np.pi / (n - 1)
    array = []
    val = 0
    for i in range(n):
        array.append(val)
        val += interval
    return np.array(array)

def gen_array_2(n):
    """Generate an array of n values equally spaced over the
    interval [0,2*Pi] using numpy's linspace.
    """
    return np.linspace(0, 2*np.pi, n)

def plot_res(n):
    """Plot the values as a function of index number.
    """
    x = np.array([i for i in range(n)])
    y = gen_array_2(n)
    plt.plot(x, y, 'o')
    plt.show()

#function definitions for problem 3
# 3a
def scalar_mult(u, c):
    """Compute scaler multiplication of an array and a number.
    Args: an np.array u, a number c
    Returns: an array 
    """
    n = u.shape[0]
    out = np.zeros(n)
    for i in range(n):
        out[i] = u[i] * c
    return out

# 3b
def component_add(a, b):
    """Compute conponent wise addition of two arrays.
    Args: two arrays with same length
    Returns: an numpy array with the same length with input arrays
    """
    n = a.shape[0]
    out = np.zeros(n)
    for i in range(n):
        out[i] = a[i] + b[i]
    return out

# 3c
def linear_function(c, x, d):
    """Evaluate the linear function y = c * x + d
    Args: c,x,d: arrays with the same length
    Returns: the result of the evaluation, an numpy array.
    """
    return component_add(scalar_mult(x, c), d)

# 3d
def component_mult(a, b):
    """Compute the component wise multiplication of two arrays
    Args: two arrays with same length
    Returns: an numpy array
    """
    n = a.shape[0]
    out = np.zeros(n)
    for i in range(n):
        out[i] = a[i] * b[i]
    return out

# 3e
def inner(a, b):
    """Compute the inner product of two arrays
    Args: two arrays with same length
    Returns: a float number
    """
    n = a.shape[0]
    out = 0
    for i in range(n):
        out += a[i] * b[i]
    return out

# 3f
def norm(a):
    """Compute the L2 norm of input array
    Args: an array a
    Return: a float number
    """
    n = a.shape[0]
    out = 0
    for i in range(n):
        out += a[i] * a[i]
    return np.sqrt(out)


#function definitions for problem 4
# 4a
@cuda.jit
def scalar_mult_kernel(d_out, d_u, d_c):
    """Kernel fucntion to do scalar multiplication"""
    i = cuda.grid(1)
    n = d_u.shape[0]
    if i >= n:
        return 
    d_out[i] = d_u[i] * d_c

def nu_scalar_mult(u, c):
    """Wrapper to do scalar multiplication
    Args: a np array and a scalar number
    Returns: result of the scalar multiplication
    """
    n = u.shape[0]
    d_u = cuda.to_device(u)
    # d_c = cuda.to_device(c)
    d_c = c
    d_out = cuda.device_array(n)
    blocks = (n + TPB - 1) // TPB
    threads = TPB
    scalar_mult_kernel[blocks, threads](d_out, d_u, d_c)
    return d_out.copy_to_host()


# 4b
@cuda.jit
def component_add_kernel(d_out, d_u, d_v):
    """Kernel function to do component wise add"""
    i = cuda.grid(1)
    n = d_u.shape[0]
    if i >= n:
        return 
    d_out[i] = d_u[i] + d_v[i]

def nu_component_add(u, v):
    """Wrapper function to do component wise add.
    Args: two numpy arrays
    Returns: a numpy array, the sum of inputs
    """
    n = u.shape[0]
    d_u = cuda.to_device(u)
    d_v = cuda.to_device(v)
    d_out = cuda.device_array(n)
    blocks = (n + TPB - 1) // TPB
    threads = TPB
    component_add_kernel[blocks, threads](d_out, d_u, d_v)
    return d_out.copy_to_host()


# 4c

@cuda.jit
def linear_function_kernel(d_out, d_c, d_x, d_d):
    """Kernel function the calculate linear function"""
    i = cuda.grid(1)
    n = d_x.shape[0]
    if i >= n:
        return 
    d_out[i] = d_c * d_x[i] + d_d[i]

def nu_linear_function(c, x, d):
    """Wrapper function to calculate linear function.
    Args: a scaler, a numpy array, another numpy array as bias
    Return: a numpy array.
    """
    n = x.shape[0]
    d_c = c
    d_x = cuda.to_device(x)
    d_d = cuda.to_device(d)
    d_out = cuda.device_array(n)
    blocks = (n + TPB - 1) // TPB
    threads = TPB
    linear_function_kernel[blocks, threads](d_out, d_c, d_x, d_d)
    return d_out.copy_to_host()


# 4d 
@cuda.jit
def component_mult_kernel(d_out, d_u, d_v):
    """Kernel function to do component wise multiplication."""
    i = cuda.grid(1)
    n = d_u.shape[0]
    if i >= n:
        return 
    d_out[i] = d_u[i] * d_v[i]

def nu_component_mult(u ,v):
    """Wrapper function to do component wise multiplication.
    Args: two numpy arrays
    Return: the result, a numpy array
    """
    n = u.shape[0]
    d_u = cuda.to_device(u)
    d_v = cuda.to_device(v)
    d_out = cuda.device_array(n)
    blocks = (n + TPB - 1) // TPB
    threads = TPB
    component_mult_kernel[blocks, threads](d_out, d_u, d_v)
    return d_out.copy_to_host()

# 4e
@cuda.jit
def inner_kernel(d_accum, d_u, d_v):
    """Kernel function to do inner product."""
    n = d_u.shape[0]
    i = cuda.grid(1)
    if i >= n:
        return 
    # accumulate the component wise product
    cuda.atomic.add(d_accum, 0, d_u[i] * d_v[i])

def nu_inner(u, v):
    """Wrapper function to do inner product.
    Args: two numpy arrays
    Returns: a float number.
    """
    n = u.shape[0]
    accum = np.zeros(1)
    d_accum = cuda.to_device(accum)
    d_u = cuda.to_device(u)
    d_v = cuda.to_device(v)
    blocks = (n + TPB - 1) // TPB
    threads = TPB
    inner_kernel[blocks, threads](d_accum, d_u, d_v)
    accum = d_accum.copy_to_host()
    return accum[0]



# 4f
@cuda.jit
def norm_kernel(d_accum, d_u):
    """Kernel function to calculate norm."""
    n = d_u.shape[0]
    i = cuda.grid(1)
    if i >= n:
        return 
    # accumulate the component wise product
    cuda.atomic.add(d_accum, 0, d_u[i] * d_u[i])

def nu_norm(u):
    """Wrapper function to calculate norm.
    Args: a numpy array
    Return: a float number"""
    n = u.shape[0]
    accum = np.zeros(1)
    d_accum = cuda.to_device(accum)
    d_u = cuda.to_device(u)
    blocks = (n + TPB - 1) // TPB
    threads = TPB
    norm_kernel[blocks, threads](d_accum, d_u)
    accum = d_accum.copy_to_host()
    return np.sqrt(accum[0])


# function definitions for problem 5
# 5.a.v reverse dot
# alternative version of inner kernel: calculate sum serially
@cuda.jit
def inner_kernel_serial(d_uv, d_u, d_v):
    """Kernel function to do inner product.
    Returns a numpy array: the component wise multiplication of
    two arrays
    """
    n = d_u.shape[0]
    i = cuda.grid(1)
    if i >= n:
        return
    d_uv[i] = d_u[i] * d_v[i]

# sum the contributions in serial order
def inner_diff(u, v):
    """Calculate the difference between two ways of accumlation 
    when doing inner product.
    Args: two numpy arrays.
    Returns: a float number, the inner product.
    """
    n = u.shape[0]
    d_u = cuda.to_device(u)
    d_v = cuda.to_device(v)
    uv = np.zeros(n)
    d_uv = cuda.to_device(uv)
    blocks = (n + TPB - 1) // TPB
    threads = TPB
    inner_kernel_serial[blocks, threads](d_uv, d_u, d_v)
    uv = d_uv.copy_to_host()
    sum_s = 0
    sum_r = 0
    for i in range(n):
        # accumulate components from head to tail
        sum_s += uv[i]
        # accumulate components from tail to head
        sum_r += uv[n-i-1]
    print("serial dot: ", sum_s)
    print("reverse dot: ", sum_r)
    return np.abs(sum_s - sum_r)


def main():
    N_2 = 11
    # Problem 2
    print("Problem 2")
    print("2.1")
    print("N = ", N_2, ": ", gen_array_1(N_2))
    print()
    print("2.2")
    print("N = ", N_2, ": ", gen_array_2(N_2))
    print()
    print("2.3")
    print("See Figure 1.")
    print()

    # Problem 3
    u = np.ones(5)
    v = np.ones(5)
    c = 2
    d = 3
    print("Problem 3")
    print("u = ", u)
    print("v = ", v)
    print("c = ", c)
    print("d = ", d)
    print("scalar mult: ", scalar_mult(u, c))
    print("component add: ", component_add(u, v))
    print("component mult: ", component_mult(u,v))
    print("linear function: ", linear_function(c, u, v))
    print("inner: ", inner(u, v))
    print("norm: ", norm(u))
    print()

    # Problem 4
    print("Problem 4")
    print("u = ", u)
    print("v = ", v)
    print("c = ", c)
    print("d = ", d)
    print("scalar mult: ", nu_scalar_mult(u, c))
    print("component add: ", nu_component_add(u, v))
    print("component mult: ", nu_component_mult(u,v))
    print("linear function: ", nu_linear_function(c, u, v))
    print("inner: ", nu_inner(u, v))
    print("norm: ", nu_norm(u))
    print()

    # Problem 5
    N_5 = 5
    print("Problem 5")
    print("N = ", N_5)
    v = np.ones(N_5)
    u = np.ones(N_5)
    for i in range(1, N_5):
        u[i] = 1 / (N_5 - 1)
    print("5.1")
    print("v = ", v)
    print()
    print("5.2")
    print("u = ", u)
    print()
    print("5.3")
    z = nu_scalar_mult(u, -1)
    # print("z = -u = ", z)
    print("norm(u + z) = ", nu_norm(nu_component_add(u, z)))
    print()
    print("5.4")
    print("u dot v = ", nu_inner(u, v))
    print()
    print("5.5")
    # print("u reverse dot v = ")
    print("Dot result difference: ", inner_diff(u, v))
    print()
    print("I repeat 5.5 for larger Ns, here are the result:")
    print("N = 1e4\nserial dot:  2.0\nreverse dot:  2.0\nDot result difference:  9.01723140601e-13")
    print()
    print("N = 1e5\nserial dot:  2.0\nreverse dot:  2.0\nDot result difference:  2.2226664953e-12")
    print()
    print("N = 1e6\nserial dot:  2.00000000001\nreverse dot:  2.00000000001\nDot result difference:  4.08117983852e-13")
    print()
    print("N = 1e7\nserial dot:  2.0000000005\nreverse dot:  2.00000000023\nDot result difference:  2.74033240544e-10")
    print()
    print("N = 1e8\nserial dot:  2.00000000613\nreverse dot:  1.99999999776\nDot result difference:  8.36787017455e-09")
    print()
    print("When N increase to 1e6, the order of summation starts to influnce results.\nThis is because of the float number round off error. The data type is 'np.float64',\nso the difference is smaller than the 'np.float32' case.")
    
    # plot for problem 2.3
    plot_res(N_2)
    plt.close()

if __name__ == '__main__':
	main()












