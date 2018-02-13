import numpy as np
import math
from numba import jit, cuda, float32, int32

'''Create Global Variable that is the block size'''
TPB = 128
RAD = 1
'''Create Global Variable that is the length of the shared memory array'''
NSHARED = 130 # This value must agree with TPB + 2*RAD

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

#@cuda.jit("void(float32[:],float32[:],float32[:])")
@cuda.jit
def derivativeKernel(d_deriv, d_f, stencil):
	sh_f = cuda.shared.array(NSHARED, dtype = float32)     
	i = cuda.grid(1)
	n = d_f.shape[0]
	if i>=n:
		return


	''' thread index and index into static shared output array'''
	tIdx = cuda.threadIdx.x
	''' index in shared input array'''
	shIdx = tIdx + RAD

	#Load regular cells
	sh_f[shIdx] = d_f[i]

	#Halo cells- Check that the entries to be loaded are within array bounds
	if tIdx < RAD:
		if i >= RAD:
			sh_f[shIdx - RAD] = d_f[i-RAD]
		if i + cuda.blockDim.x < n:
			sh_f[shIdx + cuda.blockDim.x] = d_f[i + cuda.blockDim.x]

	cuda.syncthreads()
	# Only write values where the full stencil is "in bounds"
	if i >= RAD and i < n-RAD:
		tmp =  sh_f[shIdx  ]*stencil[RAD]
		for d in range(1,RAD+1):
			tmp += sh_f[shIdx-d]*stencil[RAD-d] + sh_f[shIdx+d]*stencil[RAD+d]
		d_deriv[i] = tmp
		'''
		d_deriv[i]= sh_f[shIdx-1]*stencil[0] + \
					sh_f[shIdx  ]*stencil[1] + \
					sh_f[shIdx+1]*stencil[2]
					'''
	#RECODE THIS AS LOOP SUM FOR GENERAL STENCIL SIZE
def derivativeArray(f, order):
	n = f.shape[0]
	if order == 1:
		stencil =(n-1)/2. * np.array([-1., 0., 1.], dtype = np.float32)
	elif order == 2:
		stencil = (n-1)*(n-1)* np.array([1., -2., 1.], dtype = np.float32)
	d_f = cuda.to_device(f)
	d_deriv = cuda.device_array(n, dtype = np.float32)
	derivativeKernel[(n+TPB-1)/TPB, TPB](d_deriv, d_f, stencil)
	return d_deriv.copy_to_host()