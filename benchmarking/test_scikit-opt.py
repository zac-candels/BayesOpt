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

def run_simulation(x_coord: float, y_coord: float) -> float:
    """
    1) Write parameters to input.txt
    2) Run C++ simulation binary (./run.exe)
    3) Extract output from C++ program
    4) Return this value
    """
    # 1) write input file
    with open('input.txt', 'w') as f:
        f.write(f"x_coord={x_coord:.6f}\n")
        f.write(f"y_coord={y_coord:.6f}\n")

    # 2) run C++ simulation
    sim = subprocess.run(['./run.exe'], capture_output=True, text=True)
    if sim.returncode != 0:
        raise RuntimeError(f"Simulation error: {sim.stderr}")

    # 3) read the value in output.dat
    
    with open('./data/output.dat', 'r') as f:
        line = f.readline()
        val = float(line.strip())

    # 4) parse output (assume Analysis.py prints a single float)
    return -val


# --- scikit-optimize libraries ---
from skopt import gp_minimize
from skopt.space import Real

# Search space
search_space = [
    Real(-5, 5, name='x'),
    Real(-5, 5, name='y')
]

def objective_sk(params):
    x, y = params
    try:
        val = run_simulation(x, y)
    except Exception as e:
        print(f"Error at {params}: {e}")
        return 1e6
    return val  # minimize negative


def run_skopt():
    n_calls = 100 # Number of function evaluations
    
    xi=0.01 # controls exploration vs exploitation
    # Default value of xi is 0.01
    # Larger values of xi result in more exploration
    result = gp_minimize(func=objective_sk, dimensions=search_space,
                         acq_func='EI', n_initial_points=10,
                         n_calls=n_calls, random_state=42,
                         xi=xi)
    x, y = result.x
    print("== skopt best ==")
    print(f"x={x:.2e}, y={y:.2e}")
    print(f"Max value={-result.fun:.4f}")
    print("Number of iterations", n_calls)
    print("xi =", xi)


# --- Main entry point ---
if __name__ == '__main__':
        run_skopt()
