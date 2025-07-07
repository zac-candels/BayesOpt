"""
Workflow per evaluation:
 1. Write `input.txt` with current parameters.
 2. Run C++ simulation (e.g. `./run.exe`).
 3. Read in value output by the c++ function
 4. Parse and return objective.

Includes two variants:
  - `bayes_opt` (bayesian-optimization package)
  - `skopt` (`scikit-optimize` package)
"""
import subprocess
import numpy as np
import os

# --- Utility: run one simulation & analysis ---

def run_simulation(theta: float, nbPost: float) -> float:
    """
    1) Write parameters to input.txt
    2) Run C++ simulation binary (./run.exe)
    3) Run Analysis.py to extract hysteresis
    4) Return hysteresis (higher = better)
    """
    
    path_to_input_cpp1 = "/home/zcandels/Projects/LBM-main/examples"
    path_to_input_cpp2 = "/binary/superhydrophobic"
    
    full_path = path_to_input_cpp1 + path_to_input_cpp2
    
    
    with open(full_path + '/input.txt', 'r') as f:
        lines = f.readlines()

    # Update the required fields
    new_lines = []
    for line in lines:
        if line.strip().startswith("theta="):
            new_lines.append(f"theta={theta:.2f} #contact angle\n")
        elif line.strip().startswith("nbPost="):
            new_lines.append(f"nbPost={int(nbPost)} #number of posts in the x direction\n")
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

    # # 3) read the value in output.dat
    
    # with open('./data/output.dat', 'r') as f:
    #     line = f.readline()
    #     val = float(line.strip())
    
    # 3) run Python analysis
    analysis = subprocess.run(
    ['python', 'Analysis.py'],      # just the script name
    cwd=full_path,                  # <-- now datadir="data/" lives here
    capture_output=True,
    text=True,
    timeout=60
    )
    analysis.check_returncode()
    if analysis.returncode != 0:
        raise RuntimeError(f"Analysis error: {analysis.stderr}")

  
    return -float(analysis.stdout.strip())



from skopt import gp_minimize
from skopt.space import Real

# Search space
search_space = [
    Real(10, 150, name='theta'),
    Real(2, 50, name='nbPost')
]

def objective_sk(params):
    theta, nbPost = params
    try:
        val = run_simulation(theta, nbPost)
    except Exception as e:
        print(f"Error at {params}: {e}")
        return 1e6
    return val  


def run_skopt():
    result = gp_minimize(func=objective_sk, dimensions=search_space,
                         acq_func='EI', n_initial_points=5, n_calls=120, random_state=42)
    theta, nbPost = result.x
    print("== skopt best ==")
    print(f"theta={theta:.2e}, nbPost={nbPost:.2e}")
    print(f"Max value={-result.fun:.4f}")


if __name__ == '__main__':
        run_skopt()
