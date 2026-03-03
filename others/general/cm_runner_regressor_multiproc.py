import subprocess
from multiprocessing import Pool
import sys
import time
import math
import numpy as np

def loguniform_by_random(low, high, size) -> list:
    ln_low = math.log(low)
    ln_high = math.log(high)
    power_samples = np.random.uniform(low=ln_low, high=ln_high, size=size)
    samples = [math.exp(s) for s in power_samples]
    return samples

def run_simulation(params) -> None:

    i, gamma, reg_param = params 

    path = "/work/x5bai/project/Code_Files/general/cm_iter_regressor.py"
    envr = sys.executable
    subprocess.run([envr, path, str(i), str(gamma), str(reg_param)])

if __name__ == "__main__": 

    start_time = time.time()

    gamma_values = loguniform_by_random(1, 1000, 1000) 
    reg_param_values = loguniform_by_random(1, 100, 1000) 
    indices = list(range(1000))
    param_list = [(i, g, r) for i, g, r in zip(indices, gamma_values, reg_param_values)]  

    with Pool(processes=10) as pool:
        results = pool.map(run_simulation, param_list)

    # check the training duration
    end_time = time.time()
    duration = (end_time - start_time)/(60*60)
    print(f"Computing time: {duration:.3f} hours")