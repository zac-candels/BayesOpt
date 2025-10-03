import numpy as np
from ax.service.managed_loop import optimize

def fn(x_coord, y_coord):
    # write input file
    with open('input.txt', 'w') as f:
        f.write(f"x_coord={x_coord:.6f}\n")
        f.write(f"y_coord={y_coord:.6f}\n")

    # run C++ program
    sim = subprocess.run(['./run.exe'], capture_output=True, text=True)
    if sim.returncode != 0:
        raise RuntimeError(f"Simulation error: {sim.stderr}")

    # read result
    with open('./data/output.dat', 'r') as f:
        val = float(f.readline().strip())

    return val

best_parameters, values, experiment = optimize(
    parameters=[
        {"name": "x_coord", "type": "range", "bounds": [-5.0, 5.0]},
        {"name": "y_coord", "type": "range", "bounds": [-5.0, 5.0]},
    ],
    evaluation_function=fn,
    total_trials=10,
)

print("Best parameters:", best_parameters)
print("Values:", values)
