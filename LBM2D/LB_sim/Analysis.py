import os, math, re, sys
import numpy as np
import struct
from matplotlib import pyplot as plt

# Directory where data is stored
datadir = "data/"

# Open the Header file
HeaderFile = open(datadir+"Header.mat", 'rb')

# Simulation size, struct.unpack reads 4 bytes from Header.mat and interprets them as an integer
LX = struct.unpack('=i', HeaderFile.read(4))[0]
# Read the next 4 bytes...
LY = struct.unpack('=i', HeaderFile.read(4))[0]
LZ = struct.unpack('=i', HeaderFile.read(4))[0]

# 2D or 3D
ndim = struct.unpack('=i', HeaderFile.read(4))[0]

# What time to start at, what time to end at, and the time increment
tstart = 0
tend = struct.unpack('=i', HeaderFile.read(4))[0]
tinc = struct.unpack('=i', HeaderFile.read(4))[0]

# Slice in the z direction to plot
zslice=0

# Where to save the plots
outDirName = "figures"
os.system("mkdir -p %s"%outDirName)

for t in range(tstart, tend + 1, tinc):

    # File containing boundary ids
    file_name = datadir + "BoundaryLabels_t%li.mat"%t
    FileSolid = open(file_name, 'rb')
    dat=FileSolid.read()
    # Fill a numpy array of dimensions (LX,LY,LZ) with the data from the file in the format '=i' (integer). (4*LY*LZ,4*LZ,4) are steps taken in bytes for each dimension. E.g in the z direction we move 4 bytes to the next z value, in the y direction we move 4 bytes * the number of z values to the next y value, etc.
    solid = np.ndarray((LX, LY, LZ), '=i', dat, 0, (4 * LY * LZ, 4 * LZ, 4))
    FileSolid.close()

    file_name = datadir + "OrderParameter_t%li.mat"%t
    FileC = open(file_name, 'rb')
    dat = FileC.read()
    C = np.ndarray((LX, LY, LZ), '=d', dat, 0, (8 * LY * LZ, 8 * LZ, 8))
    liquid = np.array(C[:,:])
    # Set order parameter in the solid to 0.5 for visualisation
    liquid[np.where(np.logical_or(solid == 1, solid == -1))[0], np.where(np.logical_or(solid == 1, solid == -1))[1], np.where(np.logical_or(solid == 1, solid == -1))[2]] = 0.5
    liquid[np.where(np.logical_or(solid == 3, solid == 2))[0], np.where(np.logical_or(solid == 3, solid == 2))[1], np.where(np.logical_or(solid == 3, solid == 2))[2]] = 0.5
    FileC.close()

    file_name = datadir + "ViscousDissipation_t%li.mat"%t
    FileD = open(file_name, 'rb')
    dat = FileD.read()
    D = np.ndarray((LX, LY, LZ), '=d', dat, 0, (8 * LY * LZ, 8 * LZ, 8))
    dissipation = np.array(D[:,:])
    # Set order parameter in the solid to 0.5 for visualisation
    dissipation[np.where(np.logical_or(solid == 1, solid == -1))[0], np.where(np.logical_or(solid == 1, solid == -1))[1], np.where(np.logical_or(solid == 1, solid == -1))[2]] = 0
    dissipation[np.where(np.logical_or(solid == 3, solid == 2))[0], np.where(np.logical_or(solid == 3, solid == 2))[1], np.where(np.logical_or(solid == 3, solid == 2))[2]] = 0
    FileD.close()

    file_name = datadir + "Velocity_t%li.mat"%t
    FileV = open(file_name, 'rb')
    dat = FileV.read()
    v = np.ndarray((LX, LY, LZ, ndim), '=d', dat, 0, (ndim * 8 * LY * LZ, ndim * 8 * LZ, ndim * 8, 8))
    FileV.close()
    print("max v: ",np.amax(v))
    print(file_name)

    # Make a figure
    fig, ax = plt.subplots(2, 1, figsize = (6, 5))
    output = "%s/component_plot_%012d.png"%(outDirName, t)

    # Array to be visualised
    rgbv = np.zeros((LY, LX))
    rgbdiss = np.zeros((LY, LX))
    rgbv[:, :] = np.flip(liquid[:, :, zslice]).transpose()
    rgbdiss[:, :] = np.flip(dissipation[:, :, zslice]).transpose()
    
    # Visualise the array
    im = ax[0].imshow(rgbv,interpolation='nearest',origin='upper')
    im2 = ax[1].imshow(rgbdiss,interpolation='nearest',origin='upper')

    stepx = 1
    stepy = 1
    X, Z = np.meshgrid(np.linspace(0, LX - 1, int((LX) / stepx)), np.linspace(0, LY - 1, int((LY) / stepy)))
    # Plot velocity arrows
    ax[0].quiver(X.T, Z.T, np.flip(np.flip(-v[0:LX:stepx, 0:LY:stepy, zslice,0], 0), 1),np.flip(v[0:LX:stepx, 0:LY:stepy,zslice, 1]), width=0.0008, headwidth=7.5, headlength=7.5)

    # Color bar
    fig.colorbar(im2,orientation="horizontal")

    plt.savefig(output, dpi=400, format='png')
    plt.close(fig)
