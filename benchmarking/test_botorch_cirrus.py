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

import warnings
warnings.filterwarnings("ignore")

# Define search space bounds
bounds = torch.tensor([[-5.0, -5.0], [5.0, 5.0]])

input_tf = Normalize(
    d=2,                        # dimension of input
    bounds=bounds )   


def postProcess(data_path: str):
    with open(data_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
        if not lines:
            raise ValueError(f"No data found in {filepath}")
        return float(lines[-1])     

# Define the objective function
def objective(X: torch.Tensor) -> torch.Tensor:
    """
    X: (batch_size x 2) tensor of x,y coordinates
    returns: (batch_size x 1) tensor of -simulation_value
    """
    results = []
    jobIDs = []
    for x in X:
        # extract scalar floats
        x0 = float(x[0].item())
        x1 = float(x[1].item())

        # Create new directory where we will place updated input file, executable and slurm file
        runDirName = "run_x0" + str(x0) + "_x1" + str(x1)
        os.mkdir(runDirName)

        # Create copy of original input file and place it in the directory ./runDirName
        newInputName = runDirName  + "/input_x0" + str(x0) + "_x1" + str(x1) + ".txt"
        shutil.copy('./input.txt', newInputName)

        # Create copy of executable and place it in ./runDirName
        newExecutableName = "./" + runDirName + "/run.exe"
        shutil.copy("./run.exe", newExecutableName)

        # Create copy of slurm file and place it in ./runDirName
        newSlurmName = "./" + runDirName + "/submit.slurm"
        shutil.copy("./submit.slurm", newSlurmName)


        # Modify original input file with new parameters
        with open('input.txt', 'w') as f:
            f.write(f"x_coord={x0:.6f}\n")
            f.write(f"y_coord={x1:.6f}\n")

        # 2) run C++ simulation
        submit = subprocess.run(
            ["sbatch", "submit.slurm"],
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
        data_path = "./data/data_x" + "{:.6f}".format(X[i][0]) + "_y" + "{:.6f}".format(X[i][1]) + "/output.dat" 
        val = postProcess(data_path)

        # store the *negative* of the objective
        results.append(-val)

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
    print(f"Iteration {i+1}, Best observed value: {Y.max().item():.4f}")
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



