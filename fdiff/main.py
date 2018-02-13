import numpy as np
import matplotlib.pyplot as plt
import math

N = 512

def main():
	x = np.linspace(0, 1, N, endpoint=True, dtype=np.float32)
	
	from shared import pArray, derivativeArray
	f = pArray(x)
	print f
	plt.plot(x, f, 'o', label='f(x) = x**2')

	dfdx = derivativeArray(f, 1)
	print dfdx
	plt.plot(x, dfdx, 'o', label='First derivative')
	
	d2fdx2 = derivativeArray(f, 2)
	print d2fdx2
	plt.plot(x, d2fdx2, 'o', label='Second derivative')
	
	plt.legend()
	plt.show()

if __name__ == '__main__':
	main()