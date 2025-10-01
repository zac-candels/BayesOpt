import numpy as np
from tvtk.api import tvtk, write_data

file_name = "./data/Density_t2000.mat"

LX=100
LY=100
LZ=100
#LX=139
#LY=139
#LZ=139
#LX=140
#LY=100
#LZ=120

File = open(file_name, 'rb')
dat=File.read()
solid = np.ndarray((LX, LY, LZ), '=d', dat, 0, (8* LY * LZ, 8 * LZ, 8))
#solid[solid==-1]=1

#rho=np.concatenate((rho,rho),axis=0)
#rho=np.concatenate((rho,rho),axis=1)
#rho=np.concatenate((rho,rho),axis=2)

grid = tvtk.ImageData(origin=(0,0,0), #spacing=(10, 5, -10)
                      dimensions=solid.shape)
grid.point_data.scalars = solid.ravel(order='F')
grid.point_data.scalars.name = file_name

# Writes legacy ".vtk" format if filename ends with "vtk", otherwise
# this will write data using the newer xml-based format.
write_data(grid, '7big.vtk')