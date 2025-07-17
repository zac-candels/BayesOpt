import subprocess
import re
import os

# Set working directory
WORKDIR = "/home/zcandels/Projects/LBM-main/examples/binary/superhydrophobic/"
INPUT_FILE = os.path.join(WORKDIR, "input.txt")
RUN_CMD = "./run.exe"
ANALYSIS_CMD = ["python3", "Analysis.py"]

def run_simulation():
    print("Running C++ simulation...")
    subprocess.run(RUN_CMD, cwd=WORKDIR, check=True)

def run_analysis():
    print("Running Analysis.py...")
    result = subprocess.run(ANALYSIS_CMD, cwd=WORKDIR, capture_output=True, text=True)
    return result.stdout

def extract_last_max_velocity(output):
    max_v_matches = re.findall(r"max v:\s*([0-9.+eE-]+)", output)
    if not max_v_matches:
        raise ValueError("No 'max v:' found in Analysis.py output")
    return float(max_v_matches[-1])

def update_input_file(new_postheight):
    with open(INPUT_FILE, 'r') as f:
        lines = f.readlines()

    with open(INPUT_FILE, 'w') as f:
        for line in lines:
            if line.strip().startswith("postheight="):
                f.write(f"postheight={new_postheight:.10f} #height of the posts\n")
            else:
                f.write(line)

def main():
    # Optional: Initialize input.txt with starting values (customize if needed)

    for iteration in range(10):
        print(f"\n--- Iteration {iteration + 1} ---")
        
        # Step 1: Run the C++ simulation
        run_simulation()

        # Step 2: Run the analysis script
        output = run_analysis()
        print("Analysis Output:\n", output)

        # Step 3: Extract final max v
        alpha_final = extract_last_max_velocity(output)
        print(f"Final max v = {alpha_final}")

        # Step 4: Compute new postheight
        new_postheight = alpha_final * 10000
        print(f"Updating postheight to {new_postheight}")
        update_input_file(new_postheight)

if __name__ == "__main__":
    main()
