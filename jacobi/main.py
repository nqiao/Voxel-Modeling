import numpy as np
import matplotlib.pyplot as plt
from math import sin, sinh
from shared import update
N = 8
PI = 3.14159

def main():
	u = np.zeros(shape=[N,N], dtype=np.float32)
	for i in range(N):
		u[i,N-1]= sin(i*PI/(N-1))
	print(u)
	u = update(u,8*N)

	exact = np.zeros(shape=[N,N], dtype=np.float32)
	for i in range(N):
		for j in range(N):
			exact[i,j]= sin(i*PI/(N-1)) * sinh(j*PI/(N-1))/sinh(PI)
	error = np.max(np.abs(u-exact))
	print(error)

	coords = np.linspace(0., 1.0, N, endpoint=True)
	X,Y = np.meshgrid(coords, coords)
	plt.contour(X,Y,u, levels = [0.25, 0.5, 0.75])
	plt.contour(X,Y,exact, levels = [0.25, 0.5, 0.75])
	plt.axis([0,1,0,1])
	plt.show()

if __name__ == '__main__':
	main()