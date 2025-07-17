import numpy as np
from ax.api.client import Client
from ax.api.configs import RangeParameterConfig
import subprocess
import os

client = Client()

# Define two float parameters x_coord, y_coord
# for the function to be optimized
parameters = [
    RangeParameterConfig(
        name="x_coord", parameter_type="float", bounds=(-5, 5) ),
    RangeParameterConfig(
        name="y_coord", parameter_type="float", bounds=(-5, 5) ),
    ]

client.configure_experiment(parameters=parameters)

metric_name = "fn" # Name is used during optimization loop
objective = f"{metric_name}" 

client.configure_optimization(objective=objective)

def fn(x_coord, y_coord):
    
    # 1) write input file
    with open('input.txt', 'w') as f:
        f.write(f"x_coord={x_coord:.6f}\n")
        f.write(f"y_coord={y_coord:.6f}\n")
        
    # 2) Run C++ program
    sim = subprocess.run(['./run.exe'], capture_output=True, text=True)
    if sim.returncode != 0:
        raise RuntimeError(f"Simulation error: {sim.stderr}")
        
    # 3) Read the value in output.dat
    with open('./data/output.dat', 'r') as f:
        line = f.readline()
        val = float(line.strip())
        
    return val

for _ in range(10):
    
    trials = client.get_next_trials(max_trials=3)
    
    for trial_index, parameters in trials.items():
        x_coord = parameters["x_coord"]
        y_coord = parameters["y_coord"]
        
        result = fn(x_coord, y_coord)
        
        # Set raw_data as a dictionary with metric names as keys
        # and results as values
        raw_data = {metric_name: result}
        
        client.complete_trial(trial_index=trial_index, raw_data=raw_data)
        
        print(f"completed trial {trial_index} with {raw_data=}")
