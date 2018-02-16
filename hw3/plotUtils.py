'''
utilies for creating 2d and 3d plots based on 2d and 3d numpy arrays
'''
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure

# Create contourplot based on 2d array fvals 
#   sampled on grid with coords determined by xvals and yvals
# note that fvals is 3rd argument in previous version
def arraycontourplot(fvals, xvals, yvals, levels=[-1000,0], vars=['x','y'], 
	titlestring='', filled=False):
    fig = plt.figure()
    X,Y = np.meshgrid(xvals,yvals)
    if filled==True:
        cp = plt.contourf(X, Y, fvals, levels) #, linestyles='dashed')
    else:
        cp = plt.contour(X, Y, fvals, levels) #, linestyles='dashed')
    # plt.clabel(cp, inline=True, fontsize=10)
    plt.title(titlestring)
    plt.xlabel(vars[0])
    plt.ylabel(vars[1])
    plt.axis('square')
    plt.show()
    return cp


# Create 3d meshplot based on 2d array fvals 
#   sampled on grid with coords determined by xvals and yvals
def plot3d(fvals, xvals, yvals, titlestring='',vars=['x','y','f']):
    fig = plt.figure()
    #ax = plt.axes(projection='3d')
    ax = fig.add_subplot(111,projection='3d')

    X,Y = np.meshgrid(xvals,yvals)
    ax.plot_wireframe(X,Y, fvals, color='black')
    ax.set_title(titlestring)
    ax.set_xlabel(vars[0])
    ax.set_ylabel(vars[1])
    ax.set_zlabel(vars[2])
    # plt.savefig(fname)
    plt.show()

# Compute a tesselation of the zero isosurface
def tesselate(fvals, xvals, yvals, zvals, dx, dy, dz):
    verts, faces = measure.marching_cubes_classic(fvals, 0.0,spacing=(1.0, 1.0, 1.0))
    print ('verts, faces = '+str(verts.size//3)+', '+str(faces.size//3))

    ndex = [0,0,0]
    frac = [0,0,0]
    verts2 = np.ndarray(shape=(verts.size//3,3), dtype=float)

    for i in range(0,verts.size//3):
        for j in range(0,3):
            ndex[j] = int(verts[i][j])
            frac[j] = verts[i][j]%1
        # not index trickiness below (with 0,1,2 reversed on right-hand side)
        verts2[i][0] = xvals[ndex[2]]+(dx)*frac[2]
        verts2[i][1] = yvals[ndex[1]]+(dy)*frac[1]
        verts2[i][2] = zvals[ndex[0]]+(dz)*frac[0]
    return tuple([verts2, faces])

def exportPLY(filename, verts2, faces):
	plyf = open(filename +'.ply', 'w')
	plyf.write( "ply\n")
	plyf.write( "format ascii 1.0\n")
	plyf.write( "comment ism.py generated\n")
	plyf.write( "element vertex " + str(verts2.size/3)+'\n')
	plyf.write( "property float x\n")
	plyf.write( "property float y\n")
	plyf.write( "property float z\n")
	plyf.write( "element face " + str(faces.size/3)+'\n')
	plyf.write( "property list uchar int vertex_indices\n")
	plyf.write( "end_header\n")
	for i in range(0,verts2.size//3):
	    plyf.write(str(verts2[i][0])+' '+str(verts2[i][1])+' '+str(verts2[i][2])+'\n')
	    
	for i in range(0,faces.size//3):
	    plyf.write('3 '+str(faces[i][0])+' '+str(faces[i][1])+' '+str(faces[i][2])+'\n') 
	plyf.close()
	# end of PLY file write

# Create 3d contourplot (and surface tesselation) based on 3d array fvals 
#   sampled on grid with coords determined by xvals, yvals, and zvals
# Note that tesselator requires inputs corresponding to grid spacings
def arraycontourplot3d(fvals, xvals, yvals, zvals, dx, dy, dz, 
	levels=[-1000,0], vars=['x','y','z'], titlestring='', filename=''):
    #compute tesselation
    verts, faces = tesselate(fvals, xvals, yvals, zvals, dx, dy, dz)
                   
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')

    mesh = Poly3DCollection(verts[faces],linewidths=0.1, alpha=0.85)
    mesh.set_edgecolor([0,0,1])
    mesh.set_facecolor([0,1,0])
    ax.add_collection3d(mesh)
    ax.set_title(titlestring)
    ax.set_xlabel(vars[0])
    ax.set_ylabel(vars[1])
    ax.set_zlabel(vars[2])
    ax.set_xlim(min(xvals),max(xvals)) 
    ax.set_ylim(min(yvals),max(yvals))  
    ax.set_zlim(min(zvals),max(zvals)) 
    plt.show()
    if filename != '':
    	print('Object exported to '+filename+'.ply')
    	exportPLY(filename, verts, faces)	
