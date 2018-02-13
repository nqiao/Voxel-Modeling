import math
import numpy as np

def parab(x0):
	return x0**2

def pArray(x):
	n = x.size
	f = np.empty(n)
	for i in range(n):
		f[i] = parab(x[i])
	return f

def derivativeArray(f, order):
	n = f.shape[0]
	if order == 1:
		stencil =(n-1)/2. * np.array([-1., 0., 1.])
		#c = (n-1) / 2.
	elif order == 2:
		stencil = (n-1)*(n-1)* np.array([1., -2., 1.])
		#c = (n-1)*(n-1)
	deriv = np.zeros(n)
	derivAlt = np.zeros(n)

	for i in range(1,n-1):
		deriv[i] = (
			f[i-1]*stencil[0]+
			f[i] * stencil[1]+
			f[i+1]*stencil[2])

	# Alternative version of stencil by loop sum
	'''
	RAD = 1
	for i in range(1,n-1):
		for di in range(-RAD, RAD+1):
			deriv[i] += f[i+di]*stencil[di+RAD]
	'''

	
	return deriv