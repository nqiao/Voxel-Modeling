import numpy as np
import matplotlib.pyplot as plt
import math

N = 64

def main():
	x = np.linspace(0, 2, N, endpoint=True)
	
	from serial import sArray
	f = sArray(x)
	plt.plot(x, f, 'o', label='serial')

	from parallel import sArray
	fpar = sArray(x)
	diff = np.absolute(f - fpar)
	maxDiff = np.max(diff)
	print maxDiff
	print "Max. Diff. = ", maxDiff
	plt.plot(x, fpar, '-', label ='parallel')
	plt.plot(x, diff, 'x', label = 'difference')

	plt.legend()
	plt.show()

if __name__ == '__main__':
	main()