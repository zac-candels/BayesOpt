import os, math, re, sys, time
import numpy as np
import struct
from matplotlib import pyplot as plt

start = time.perf_counter()

def read_params(path):
    params = {}
    # matches:   key   =   integer   (ignoring anything after a '#')
    pat = re.compile(r'^\s*([A-Za-z_]\w*)\s*=\s*([0-9]+)')
    with open(path) as f:
        for line in f:
            m = pat.match(line)
            if m:
                key, val = m.group(1), int(m.group(2))
                params[key] = val
    return params

# Usage
path = "../LBM2D/LB_sim"
path_input = path + "/input.txt"
params = read_params(path_input)
saveInterval = params["saveInterval"]
Num_steps   = params["timesteps"]

plt.close('all')
def coord_k(k, ly, lz):
    """From a k value, determines its xk, yk, and zk."""
    xk = math.floor(k/(ly*lz))
    yk = math.floor((k - xk*lz*ly)/lz)
    zk = k - xk*lz*ly - yk*lz
    return xk, yk, zk

def interpolate_x_direction(pts_x, pts_phi):
    pts_x = np.asarray(pts_x)
    pts_phi = np.asarray(pts_phi)
    gas_phase_x, gas_phase_phi = [], []
    liquid_phase_x, liquid_phase_phi = [], []
    if len(pts_x) > 2:
        #sorted_indices = np.argsort(pts_phi)
        #pts_phi = pts_phi[sorted_indices]
        #pts_x = pts_x[sorted_indices]
        mask_gas = pts_phi < 0.5
        mask_liquid = pts_phi >= 0.5
        
        gas_phase_phi = pts_phi[mask_gas]
        gas_phase_x = pts_x[mask_gas]
        
        liquid_phase_phi = pts_phi[mask_liquid]
        liquid_phase_x = pts_x[mask_liquid]
        
        if len(liquid_phase_phi) == 0:
            closest_gas_index = np.argmax(gas_phase_phi)
            closest_gas_pt = gas_phase_x[closest_gas_index]
            # Nothing to interpolate
            return closest_gas_pt
        elif len(gas_phase_phi) == 0:
            closest_liquid_index = np.argmin(liquid_phase_phi)
            closest_liquid_pt = liquid_phase_x[closest_liquid_index]
            return closest_liquid_pt
        else:
            closest_gas_index = np.argmax(gas_phase_phi)
            closest_gas_pt = gas_phase_x[closest_gas_index]
            
            closest_liquid_index = np.argmin(liquid_phase_phi)
            closest_liquid_pt = liquid_phase_x[closest_liquid_index]
        
        x0, x1 = [closest_liquid_pt, closest_gas_pt]
        phi0, phi1\
            = [gas_phase_phi[closest_gas_index],liquid_phase_phi[closest_liquid_index]]
            
    if len(pts_x) == 2:
        x0, x1 = pts_x[0], pts_x[1]
        phi0, phi1 = pts_phi[0], pts_phi[1]
    
    m = ( phi1 - phi0 )\
        /( x1 - x0 )
    x_interpolated = x0 + (1/m)*(0.5 - phi0)
    
    return x_interpolated


def interpolate_y_direction(pts_y, pts_phi):
    pts_y = np.asarray(pts_y)
    pts_phi = np.asarray(pts_phi)
    gas_phase_y, gas_phase_phi = [], []
    liquid_phase_y, liquid_phase_phi = [], []
    if len(pts_y) > 2:
        #sorted_indices = np.argsort(pts_phi)
        #pts_phi = pts_phi[sorted_indices]
        #pts_x = pts_x[sorted_indices]
        mask_gas = pts_phi < 0.5
        mask_liquid = pts_phi >= 0.5
        
        gas_phase_phi = pts_phi[mask_gas]
        gas_phase_y = pts_y[mask_gas]
        
        liquid_phase_phi = pts_phi[mask_liquid]
        liquid_phase_y = pts_y[mask_liquid]
        
        if len(liquid_phase_phi) == 0:
            closest_gas_index = np.argmax(gas_phase_phi)
            closest_gas_pt = gas_phase_y[closest_gas_index]
            # Nothing to interpolate
            return closest_gas_pt
        elif len(gas_phase_phi) == 0:
            closest_liquid_index = np.argmin(liquid_phase_phi)
            closest_liquid_pt = liquid_phase_y[closest_liquid_index]
            return closest_liquid_pt
        else:
            closest_gas_index = np.argmax(gas_phase_phi)
            closest_gas_pt = gas_phase_y[closest_gas_index]
            
            closest_liquid_index = np.argmin(liquid_phase_phi)
            closest_liquid_pt = liquid_phase_y[closest_liquid_index]
        
        y0, y1 = [closest_liquid_pt, closest_gas_pt]
        phi0, phi1\
            = [gas_phase_phi[closest_gas_index],liquid_phase_phi[closest_liquid_index]]
            
    if len(pts_y) == 2:
        y0, y1 = pts_y[0], pts_y[1]
        phi0, phi1 = pts_phi[0], pts_phi[1]
    
    m = ( phi1 - phi0 )\
        /( y1 - y0 )
    y_interpolated = y0 + (1/m)*(0.5 - phi0)
    
    return y_interpolated



plt.close('all')

# Directory where data is stored
datadir = path + "/data/"

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

def coord_k(k, ly, lz):
    """From a k value, determines its xk, yk, and zk."""
    xk = math.floor(k/(ly*lz))
    yk = math.floor((k - xk*lz*ly)/lz)
    zk = k - xk*lz*ly - yk*lz
    return xk, yk, zk
#pressure = np.zeros((tend, LX, LY, LZ))

# Slice in the z direction to plot
zslice=0

# Where to save the plots
outDirName = "figures"
os.system("mkdir -p %s"%outDirName)

t = 0

#Num_steps = int(100)
x_n = 0
centroid_pos = []
for n_idx in range(0, Num_steps, saveInterval):
    pattern_bdy = re.compile("BoundaryLabels_t" + str(n_idx) + ".mat")
    pattern_phi = re.compile("OrderParameter_t" + str(n_idx) + ".mat")

    files = os.listdir(datadir)
    max_n = -1
    max_m = -1
    target_file_bdy = None
    target_file_phi = None        
                

    # File containing boundary ids
    file_name_bdy = os.path.join(datadir, "BoundaryLabels_t" + str(n_idx) + ".mat")
    FileSolid = open(file_name_bdy, 'rb')
    dat=FileSolid.read()



    file_name_phi = os.path.join(datadir, "OrderParameter_t" + str(n_idx) + ".mat")
    File_phi = open(file_name_phi, 'rb')
    dat = File_phi.read()

    # Fill a numpy array of dimensions (LX,LY,LZ) with the data from the file in the format '=i' (integer). (4*LY*LZ,4*LZ,4) are steps taken in bytes for each dimension. E.g in the z direction we move 4 bytes to the next z value, in the y direction we move 4 bytes * the number of z values to the next y value, etc.
    solid = np.ndarray((LX, LY, LZ), '=i', dat, 0, (4 * LY * LZ, 4 * LZ, 4))
    FileSolid.close()

    phi = np.ndarray((LX, LY, LZ), '=d', dat, 0, (8 * LY * LZ, 8 * LZ, 8))
    liquid = np.array(phi[:,:])
    # Set order parameter in the solid to 0.5 for visualisation
    # liquid[np.where(np.logical_or(solid == 1, solid == -1))[0], np.where(np.logical_or(solid == 1, solid == -1))[1], np.where(np.logical_or(solid == 1, solid == -1))[2]] = 0.
    # liquid[np.where(np.logical_or(solid == 3, solid == 2))[0], np.where(np.logical_or(solid == 3, solid == 2))[1], np.where(np.logical_or(solid == 3, solid == 2))[2]] = 0.
    File_phi.close()
    phi = liquid[:,:,0]

    file_name = os.path.join(datadir, "Velocity_t" + str(n_idx) + ".mat")
    FileV = open(file_name, 'rb')
    dat = FileV.read()
    v = np.ndarray((LX, LY, LZ, ndim), '=d', dat, 0, (ndim * 8 * LY * LZ, ndim * 8 * LZ, ndim * 8, 8))
    FileV.close()


    v_x = v[:, :, 0, 0]
    v_y = v[:, :, 0, 1]
    phi_mult_vel = np.array([0.0, 0.0])
    phi_droplet = 0
        
    for i in range(len(phi[:,0])):
        for j in range(len(phi[0,:])):
            v_mag2 = v_x[i,j]**2 + v_y[i,j]**2
            if phi[i,j] > 0.5: 
                # if( v_mag2 > 0.000 ):
                #     print("\n||v||^2 = ", v_mag2) 
                phi_droplet += phi[i,j] 
                phi_mult_vel += phi[i,j] * np.array([ v_x[i,j],  v_y[i,j] ]) 

    phi_droplet = np.asarray( phi_droplet )
    phi_mult_vel = np.asarray( phi_mult_vel )

    droplet_vel = phi_mult_vel / phi_droplet 
    print("vel = ", droplet_vel[0])
                
    
        



    
    





