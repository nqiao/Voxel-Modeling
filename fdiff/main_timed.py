import numpy as np
import matplotlib.pyplot as plt
from time import time
from numba import cuda

N = 640000

def main():
	x = np.linspace(0, 1, N, endpoint=True)
	from serial import sArray
	start = time()
	f = sArray(x)
	elapsed = time() - start
	print("--- Serial timing: %s seconds ---" % elapsed)

	from parallel import sArray
	start = time()
	fpar = sArray(x)
	elapsed = time() - start
	print("--- 1st parallel timing: %s seconds ---" % elapsed)
	start = time()
	fpar = sArray(x)
	elapsed = time() - start
	print("--- 2nd parallel timing: %s seconds ---" % elapsed)

if __name__ == '__main__':
	main()