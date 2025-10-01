"""
What happens in each iteration of optimization procedure:
 1. Write `input.txt` with current parameters.
 2. Run C++ simulation (e.g. `./run.exe`).
 3. Read in value output by the c++ function
 4. Parse and return objective.
"""
import subprocess
import numpy as np
import os
import re
import time

# --- Utility: run one simulation & analysis ---

def run_simulation(theta: float, postfraction: float) -> float:
    """
    Run one simulation via SLURM, wait until it finishes, then run analysis.
    Returns: negative velocity (for minimization in gp_minimize).
    """
    full_path = "../LBM2D/LB_sim"

    # --- Step 1: Write input.txt ---
    with open(os.path.join(full_path, 'input.txt'), 'r') as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        if line.strip().startswith("theta="):
            new_lines.append(f"theta={theta:.2f} #contact angle\n")
        elif line.strip().startswith("postfraction="):
            new_lines.append(f"postfraction={postfraction:.2f} #number of posts in the x direction\n")
        else:
            new_lines.append(line)

    with open(os.path.join(full_path, 'input.txt'), 'w') as f:
        f.writelines(new_lines)

    # --- Step 2: Submit job with sbatch ---
    submit = subprocess.run(
        ["sbatch", "submit.slurm"],
        cwd=full_path,
        capture_output=True,
        text=True
    )
    submit.check_returncode()

    # Example output: "Submitted batch job 12345"
    m = re.search(r"Submitted batch job (\d+)", submit.stdout)
    if not m:
        raise RuntimeError(f"Couldn't parse job ID from sbatch output: {submit.stdout}")
    job_id = m.group(1)
    print(f"[run_simulation] Submitted job {job_id} for theta={theta:.2f}, postfraction={postfraction:.2f}")

    # --- Step 3: Wait until job completes ---
    while True:
        check = subprocess.run(
            ["squeue", "-j", job_id],
            capture_output=True,
            text=True
        )
        if job_id not in check.stdout:
            break  # job finished
        time.sleep(5)  # wait 5 seconds before checking again

    print(f"[run_simulation] Job {job_id} finished. Running analysis...")

    # --- Step 4: Run analysis script ---
    analysis = subprocess.run(
        ["python", "centroid_velocity.py"],
        capture_output=True,
        text=True,
        timeout=300
    )
    analysis.check_returncode()

    lines = analysis.stdout.strip().splitlines()
    if not lines:
        raise RuntimeError("centroid_velocity.py produced no output")

    last = lines[-1]
    m = re.search(r"vel\s*=\s*:? *([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)", last)
    if not m:
        raise RuntimeError(f"Couldn't parse a number from centroid_velocity.py output: {last!r}")

    val = float(m.group(1))
    print(f"[run_simulation] Parsed velocity = {val:.4f}")
    return -val  # minus sign because gp_minimize minimizes



from skopt import gp_minimize
from skopt.space import Real
from skopt.callbacks import VerboseCallback

def print_best_so_far(res):
    current_best_idx = int(np.argmin(res.func_vals))
    current_best_val = -res.func_vals[current_best_idx]
    current_best_params = res.x_iters[current_best_idx]
    theta, postfraction = current_best_params
    print(f"[Iter {len(res.func_vals)}] Best objective so far: {current_best_val:.4f} at (theta={theta:.2f}, postfraction={postfraction:.2f})")


# Search space
search_space = [
    Real(30, 150, name='theta'),
    Real(0.1, 0.9, name='postfraction')
]

def objective_sk(params):
    theta, postfraction = params
    try:
        val = run_simulation(theta, postfraction)
    except Exception as e:
        print(f"Error at {params}: {e}")
        return 1e6
    return val  



def run_skopt():
    result = gp_minimize(
        func=objective_sk,
        dimensions=search_space,
        acq_func='EI',
        n_initial_points=5,
        n_calls=100,
        random_state=42,
        callback=[print_best_so_far]  # <-- Use your callback here
    )
    theta, postfraction = result.x
    print("== skopt best ==")
    print(f"theta={theta:.2e}, postfraction={postfraction:.2e}")
    print(f"Max value={-result.fun:.4f}")


run_skopt()
