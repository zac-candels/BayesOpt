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

# Define search space bounds
bounds = torch.tensor([[0.0, 0.0], [150.0, 1.0]])

input_tf = Normalize(
    d=2,                        # dimension of input
    bounds=bounds )

# Define the objective function
def objective(X: torch.Tensor) -> torch.Tensor:
    """
    X: (batch_size x 2) tensor of x,y coordinates
    returns: (batch_size x 1) tensor of -simulation_value
    """
    results = []
    for x in X:
        # extract scalar floats
        theta = float(x[0].item())
        postfraction = float(x[1].item())

        path_to_input_cpp1 = "/home/zcandels/LBM/examples"
        path_to_input_cpp2 = "/binary/superhydrophobic_wellbalanced"
        
        full_path = path_to_input_cpp1 + path_to_input_cpp2
        
        
        with open(full_path + '/input.txt', 'r') as f:
            lines = f.readlines()
    
        # Update the required fields
        new_lines = []
        for line in lines:
            if line.strip().startswith("theta="):
                new_lines.append(f"theta={theta:.2f} #contact angle\n")
            elif line.strip().startswith("postfraction="):
                new_lines.append(f"postfraction={postfraction:.2f} #number of posts in the x direction\n")
            else:
                new_lines.append(line)
        
        # Write back updated input.txt
        with open(full_path + '/input.txt', 'w') as f:
            f.writelines(new_lines)
    
        # 2) run C++ simulation
        sim = subprocess.run(
        [full_path + '/run.exe'],
        capture_output=True,
        text=True,
        cwd=full_path      # <-- ensure run.exe sees the right input.txt
       )
        
        if sim.returncode != 0:
            raise RuntimeError(f"Simulation error: {sim.stderr}")
        
        # 3) run Python analysis
        analysis = subprocess.run(
        ['python', 'Analysis.py'],      # just the script name
        cwd=full_path,                  # <-- now datadir="data/" lives here
        capture_output=True,
        text=True,
        timeout=120
        )
        analysis.check_returncode()
        lines = analysis.stdout.strip().splitlines()
        if not lines:
            raise RuntimeError("Analysis.py produced no output")
        
        # Grab the last line, e.g. "max v:  3.1415"
        last = lines[-1]
        
        # Extract the floating‑point number after the colon
        m = re.search(r"max v:\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)", last)
        if not m:
            raise RuntimeError(f"Couldn't parse a number from Analysis.py output: {last!r}")
        
        val = float(m.group(1))

        results.append(val)

    # stack into a (batch_size x 1) tensor
    return torch.tensor(results, dtype=X.dtype).unsqueeze(-1)


# Set random seed for reproducibility
torch.manual_seed(42)

# Initialize with random points
n_init = 3
X = bounds[0] + (bounds[1] - bounds[0]) * torch.rand(n_init, 2)
Y = objective(X)

# Optimization loop parameters
n_iterations = 10
batch_size = 1

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
        raw_samples=200000,
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