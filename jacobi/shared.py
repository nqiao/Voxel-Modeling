import numpy as np
import math
from numba import jit, cuda, float32, int32

'''Create Global Variable that is the block size'''
TPB = 8
RAD = 1
SH_N = 10 # MUST AGREE WITH TPB+2*RAD
'''
stencil = (
	(.05, .2, .05),
	(.2, 0., .2),
	(.05, .2, .05))
See, for example, en.wikipedia.org/wiki/Discrete_Laplace_operator
The -3 in the center cancels out for Laplace's equation.
'''
#@cuda.jit("void(float32[:,:],float32[:,:])")
@cuda.jit
def updateKernel(d_v, d_u):
	#i = cuda.blockIdx.x*cuda.blockDim.x + cuda.threadIdx.x
	i,j = cuda.grid(2)
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
    # left and right
	if t_i<RAD:
		sh_u[sh_i - RAD, sh_j] = d_u[i-RAD, j]
		sh_u[sh_i + NX , sh_j] = d_u[i+NX , j]
    
    # top and bottom
	if t_j<RAD:
		sh_u[sh_i, sh_j - RAD] = d_u[i, j - RAD]
		sh_u[sh_i, sh_j + NY ] = d_u[i, j + NY ]

	#Halo corner values
	if t_i<RAD and t_j<RAD:
		#upper left
		sh_u[sh_i - RAD, sh_j - RAD] = d_u[i-RAD, j - RAD]
		sh_u[sh_i - RAD, sh_j - RAD] = d_u[i-RAD, j - RAD]
		#upper right
		sh_u[sh_i + NX, sh_j - RAD] = d_u[i + NX, j - RAD]
		sh_u[sh_i + NX, sh_j - RAD] = d_u[i + NX, j - RAD]
		#lower left
		sh_u[sh_i - RAD, sh_j + NY] = d_u[i-RAD, j + NY]
		sh_u[sh_i - RAD, sh_j + NY] = d_u[i-RAD, j + NY]
		#lower right
		sh_u[sh_i + NX, sh_j + NX] = d_u[i + NX, j + NY]
		sh_u[sh_i + NX, sh_j + NY] = d_u[i + NX, j + NY]

	cuda.syncthreads()


	#stencil = cuda.local.array(shape=(3,3), dtype = float32)
	#stencil = [[.05, .25, .05],[ .25, 0.0, .25], [.05, .25, .05]]
	'''
	if i>0 and j>0 and i<n-1 and j<n-1:
		d_v[i, j] = \
			sh_u[sh_i-1, sh_j-1]*stencil[0, 0] + \
			sh_u[sh_i, sh_j-1]*stencil[1, 0] + \
			sh_u[sh_i+1, sh_j-1]*stencil[2, 0] + \
			sh_u[sh_i-1, sh_j]*stencil[0, 1] + \
			sh_u[sh_i, sh_j]*stencil[1, 1] + \
			sh_u[sh_i+1, sh_j]*stencil[2, 1] + \
			sh_u[sh_i-1, sh_j+1]*stencil[0, 2] + \
			sh_u[sh_i, sh_j+1]*stencil[1, 2] + \
			sh_u[sh_i+1, sh_j+1]*stencil[2, 2]
			'''
	#stencil coefficient values. Reorg the eqn!?
	corner = .05
	edge = .2
	#Reverse the inequalities!? if i>=n: return
	if i>0 and j>0 and i<dims[0]-1 and j<dims[1]-1:
		d_v[i, j] = \
			sh_u[sh_i-1, sh_j -1]*corner + \
			sh_u[sh_i, sh_j -1]*edge + \
			sh_u[sh_i+1, sh_j -1]*corner + \
			sh_u[sh_i-1, sh_j]*edge + \
			sh_u[sh_i, sh_j]*0. + \
			sh_u[sh_i+1, sh_j]*edge + \
			sh_u[sh_i-1, sh_j +1]*corner + \
			sh_u[sh_i, sh_j + 1] * edge + \
			sh_u[sh_i+1, sh_j +1]*corner


def update(u, iter_count):

	''' Compute number of entries in shared array'''
	#s_memSize = (TPB + 2*RAD)*(TPB + 2*RAD)# * u.itemsize
	''' Compute memory needed for dynamic shared array'''
	#s_memSize = dist_array.itemsize*shN

	d_u = cuda.to_device(u)
	d_v = cuda.to_device(u)
	dims = u.shape
	gridSize = [(dims[0]+TPB-1)//TPB, (dims[1]+TPB-1)//TPB]
	blockSize = [TPB, TPB]

	'''Launch kernel with optional parameters specifying the stream number
	and the amount of shared memory to allocate'''
	#not using dyn shared due to lack of 2D arrays???
	for k in range(iter_count):
		updateKernel[gridSize, blockSize](d_v, d_u)
		updateKernel[gridSize, blockSize](d_u, d_v)
	'''This will output the ptx code'''
	#print derivativeKernel.ptx

	return d_u.copy_to_host()
