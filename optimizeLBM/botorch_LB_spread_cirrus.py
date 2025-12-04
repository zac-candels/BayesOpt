import torch

torch.set_default_dtype(torch.double)

torch.manual_seed(2)

from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import qLogExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
import subprocess 
import os
import re
import time
import shutil
import struct
import numpy as np

import warnings
warnings.filterwarnings("ignore")

# Define search space bounds
bounds = torch.tensor([[30.0, 0.1], [150, 0.9]])

input_tf = Normalize(
    d=2,                        # dimension of input
    bounds=bounds )   

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

def postProcess(data_path: str):

    # Usage
    path_input = data_path + "/input.txt"
    params = read_params(path_input)
    saveInterval = params["saveInterval"]
    Num_steps   = params["timesteps"]
    postHeight = params["postheight"]
    posx = params["posx"]

    n_idx = Num_steps - saveInterval

    HeaderFile = open(data_path + "/Header.mat", 'rb')
    LX = struct.unpack('=i', HeaderFile.read(4))[0]
    # Read the next 4 bytes...
    LY = struct.unpack('=i', HeaderFile.read(4))[0]
    LZ = struct.unpack('=i', HeaderFile.read(4))[0]
    # 2D or 3D
    ndim = struct.unpack('=i', HeaderFile.read(4))[0]

    pattern_bdy = re.compile("BoundaryLabels_t" + str(n_idx) + ".mat")
    pattern_phi = re.compile("OrderParameter_t" + str(n_idx) + ".mat")

    max_n = -1
    max_m = -1
    target_file_bdy = None
    target_file_phi = None        
                

    # File containing boundary ids
    file_name_bdy = os.path.join(data_path, "BoundaryLabels_t" + str(n_idx) + ".mat")
    FileSolid = open(file_name_bdy, 'rb')
    dat=FileSolid.read()



    file_name_phi = os.path.join(data_path, "OrderParameter_t" + str(n_idx) + ".mat")
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


    CL_pos = []
    for i in range(len(phi[:,0])):
        for j in range(len(phi[0,:])):
            if( i > posx ):
                if( j >= postHeight and j <= postHeight + 1):
                    if( phi[i,j] > 0.4 and phi[i,j] < 0.6 ):
                        CL_pos.append(i)
    CL_pos = np.asarray(CL_pos)
    largest_CL_vals = np.sort(CL_pos)[-10:] # Get the 10 values with the largest x-values
    CL_pos = np.mean(largest_CL_vals) - posx

    return CL_pos

    # with open(data_path, 'r') as f:
    #     lines = [line.strip() for line in f if line.strip()]
    #     if not lines:
    #         raise ValueError(f"No data found in {filepath}")
    #     return float(lines[-1])     

# Define the objective function
def objective(X: torch.Tensor) -> torch.Tensor:
    """
    X: (batch_size x 2) tensor of x,y coordinates
    returns: (batch_size x 1) tensor of -simulation_value
    """
    results = []
    jobIDs = []
    runDirs = []
    for x in X:
        # extract scalar floats
        theta = float(x[0].item())
        postFrac = float(x[1].item())

        path_to_LBM = "../LBM2D/LB_sim"
        os.makedirs(path_to_LBM + "/data", exist_ok=True)

        # Create new directory where we will place updated input file, executable and slurm file
        runDirName = path_to_LBM + "/data" + "/run_theta" + str(theta) + "_postFrac" + str(postFrac)
        runDirs.append(runDirName)
        os.makedirs(runDirName, exist_ok=True)

        # Create copy of original input file and place it in the directory ./runDirName
        newInputName = runDirName  + "/input.txt"
        shutil.copy(path_to_LBM + "/input.txt", newInputName)

        # Create copy of executable and place it in ./runDirName
        newExecutableName = runDirName + "/run.exe"
        shutil.copy(path_to_LBM + "/run.exe", newExecutableName)

        # Create copy of slurm file and place it in ./runDirName
        newSlurmName = runDirName + "/submit_cirrus.slurm"
        shutil.copy(path_to_LBM + "/submit_cirrus.slurm", newSlurmName)


        # Modify original input file with new parameters
        with open(newInputName, 'r') as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            if line.strip().startswith("theta="):
                new_lines.append(f"theta={theta:.2f} #contact angle\n")
            elif line.strip().startswith("postfraction="):
                new_lines.append(f"postfraction={postFrac:.2f} #number of posts in the x direction\n")
            else:
                new_lines.append(line)

        with open(newInputName, 'w') as f:
            f.writelines(new_lines)

        # 2) run C++ simulation
        submit = subprocess.run(
            ["sbatch", "submit_cirrus.slurm"],
            cwd=runDirName,
            capture_output=True, text=True)
        if submit.returncode != 0:
            raise RuntimeError(f"Simulation error: {submit.stderr}")
        submit.check_returncode()

        m = re.search(r"Submitted batch job (\d+)", submit.stdout)
        if not m:
            raise RuntimeError(f"Couldn't parse job ID from sbatch output: {submit.stdout}")
        jobIDs.append(m.group(1))

    # Wait until job completes
    for i in range(len(jobIDs)):
        while True:
            check = subprocess.run(
                ["squeue", "-j", jobIDs[i]],
                capture_output=True,
                text=True
            )
            if jobIDs[i] not in check.stdout:
                break
            time.sleep(5)

        # 3) read the value
        data_path = runDirs[i] 
        val = postProcess(data_path)

        # store the *negative* of the objective
        results.append(val)

    # stack into a (batch_size x 1) tensor
    return torch.tensor(results, dtype=X.dtype).unsqueeze(-1)


# Set random seed for reproducibility
torch.manual_seed(12)

# Initialize with random points
n_init = 10
X = bounds[0] + (bounds[1] - bounds[0]) * torch.rand(n_init, 2)
Y = objective(X)

# Optimization loop parameters
n_iterations = 100
batch_size = 2

# Optimization loop
for i in range(n_iterations):
    # Fit a GP model to the current data
    gp = SingleTaskGP(
    train_X=X,               # shape (n,2)
    train_Y=Y,               # shape (n,1)
    input_transform=input_tf,
    outcome_transform=Standardize(m=1),
    )
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)
    
    # Define the q-EI acquisition function
    best_f = Y.max().item()
    qEI = qLogExpectedImprovement(gp, best_f=best_f)
    
    # Optimize the acquisition function to get the next batch of points
    candidates, _ = optimize_acqf(
        qEI,
        bounds=bounds,
        q=batch_size,
        num_restarts=200,
        raw_samples=40000,
    )
    
    # Evaluate the objective at the new points
    new_Y = objective(candidates)
    
    # Update the dataset
    X = torch.cat([X, candidates], dim=0)
    Y = torch.cat([Y, new_Y], dim=0)
    
    # Print progress
    print(f"Iteration {i+1}, Best observed value: {abs(Y.max().item()):.4f}")
    best_idx = Y.argmax()
    best_X = X[best_idx]
    best_Y = Y[best_idx]
    print(f"Best point found: ({best_X[0]:.4f}, {best_X[1]:.4f})\n\n")

# Report the final result
best_idx = Y.argmax()
best_X = X[best_idx]
best_Y = Y[best_idx]
print(f"\nBest point found: ({best_X[0]:.4f}, {best_X[1]:.4f})")
print(f"Maximum value: {best_Y.item():.4f}")



