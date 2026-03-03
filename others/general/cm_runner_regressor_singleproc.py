import subprocess
import sys
import time
import math
import numpy as np
from itertools import product

def loguniform_by_random(low, high, size) -> list:
    ln_low = math.log(low)
    ln_high = math.log(high)
    power_samples = np.random.uniform(low=ln_low, high=ln_high, size=size)
    samples = [math.exp(s) for s in power_samples]
    return samples

def main() -> None:

    start_time = time.time()

    ##gamma_values = loguniform_by_random(0.1, 1000, 1000)  ### used in 2025/11/11
    ##reg_param_values = loguniform_by_random(0.01, 100, 1000)  ### used in 2025/11/11
    ##combinations = [(g, r) for g,r in zip(gamma_values, reg_param_values)]  ### 2025/11/11

    gamma_values = [0.1, 500, 1000]  ### used in 2026/02/04&06
    reg_param_values = [0.01, 50, 100]  ### used in 2026/02/04
    combinations = list(product(gamma_values, reg_param_values))*24  ### 2026/02/04&06

    ##num_iterations = 1000 ### 2025/11/11
    num_iterations = 24*len(gamma_values)*len(reg_param_values) ### 2026/02/04&06

    for i in range(num_iterations):    
        print(f"\n--- Running iteration {i} ---")

        gamma = combinations[i][0]
        reg_param = combinations[i][1]

        path = "/work/x5bai/project/Code_Files/general/cm_iter_regressor.py"
        envr = sys.executable
        subprocess.run([envr, path, str(i), str(gamma), str(reg_param)])

    print("-- All Simulations Completed --")

    # check the training duration
    end_time = time.time()
    duration = (end_time - start_time)/(60*60)
    print(f"Computing time: {duration:.3f} hours")

if __name__ == "__main__": 
    main()