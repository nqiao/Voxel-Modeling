import numpy as np
import math
from numba import jit, cuda, float32

@cuda.jit(device = True)
def parab(x0):
	return x0**2

@cuda.jit #Lazy compilation
#@cuda.jit('void(float32[:], float32[:])') #Eager compilation
def pKernel(d_f, d_x):
	i = cuda.grid(1)
	n = d_x.shape[0]	
	if i < n:
		d_f[i] = parab(d_x[i])

def pArray(x):
	TPB = 32
	n = x.shape[0]
	d_x = cuda.to_device(x)
	d_f = cuda.device_array(n, dtype = np.float32) #need dtype spec for eager compilation
	pKernel[(n+TPB-1)/TPB, TPB](d_f, d_x)
	return d_f.copy_to_host()

@cuda.jit
def derivativeKernel(d_deriv, d_f, stencil):
	i = cuda.grid(1)
	n = d_f.shape[0]	
	if i > 1 and i < n-1:
		d_deriv[i] = (
			d_f[i-1]*stencil[0]+
			d_f[i] * stencil[1]+
			d_f[i+1]*stencil[2])

def derivativeArray(f, order):
	TPB = 32 # Move to top of file!?
	n = f.shape[0]
	if order == 1:
		stencil =(n-1)/2. * np.array([-1., 0., 1.])
	elif order == 2:
		stencil = (n-1)*(n-1)* np.array([1., -2., 1.])
	d_f = cuda.to_device(f)
	d_deriv = cuda.device_array(n, dtype = np.float32)
	derivativeKernel[(n+TPB-1)/TPB, TPB](d_deriv, d_f, stencil)

	return d_deriv.copy_to_host()