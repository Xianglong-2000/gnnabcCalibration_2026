import os
import sys
sys.path.append('C:/Users/MECHREV/CellModeller-ingallslab')
import re
import itertools
import numpy as np
from datetime import datetime
import psutil
from CellModeller.Simulator import Simulator
import random
import time

import warnings
warnings.filterwarnings("ignore")

# setup for defaults
dt_default = 0.025
sim_time_default = None
max_cells_default = 140

def simulate(modfilename, reg_param, gamma, iter_index = 0, max_cells=None, max_time=None) -> None:
    (path,name) = os.path.split(modfilename)
    modname = str(name).split('.')[0]
    sys.path.append(path)

    ##dt_now = str(datetime.now())[:10]  # get today's date
    dt_now = "2025-09-25"

    """
    Change the output_dir to ../Data Files/_sim_exp_pkl_data_abc/.. for abc sim exp data.
    Change the output_dir to ../Data Files/_sim_pkl_data_gnn/.. for gnn sim data.
    """

    if iter_index < 10:  ## generate folder's path
        output_dir = "D:/Projects/GNN Research/Data Files/_sim_pkl_data_gnn/"+dt_now+f"/iteration 00{iter_index}/"+f"gamma={gamma}_reg_param={reg_param}_iter={iter_index}/data"
    elif (iter_index >= 10) and (iter_index <100):
        output_dir = "D:/Projects/GNN Research/Data Files/_sim_pkl_data_gnn/"+dt_now+f"/iteration 0{iter_index}/" + f"gamma={gamma}_reg_param={reg_param}_iter={iter_index}/data"
    elif (iter_index >= 100) and (iter_index <1000):
        output_dir = "D:/Projects/GNN Research/Data Files/_sim_pkl_data_gnn/"+dt_now+f"/iteration {iter_index}/" + f"gamma={gamma}_reg_param={reg_param}_iter={iter_index}/data"
    else:
        print("We don't want the number of iterations greater than 999 since it's too computationally heavy")

    print(output_dir)
    os.makedirs(output_dir, exist_ok=True)  # create a new folder if it doesn't exist and do nothing if it exists
    os.chdir(output_dir)  # generate the simulated data in that folder

    # Extract dt and sim_time from the simulation module
    sim_file = os.path.abspath(modfilename)   ### this could cause error, try sim_file = "C:/Users/MECHREV/CellModeller-ingallslab/Models/biophysics_calibration/"+modfilename
    (dt, sim_time) = sim_time_parameters(sim_file)

    params = {"gamma": gamma, "reg_param": reg_param}
    # print(params.values())
    sim = Simulator(modname, dt, pickleSteps=2, saveOutput = True, psweep=True, params=params)

    if max_cells:
        while len(sim.cellStates) <= max_cells:
            sim.step()

    # print out the available memory
    print(psutil.virtual_memory())


def sim_time_parameters(file_path:str):
    """
    Extract the dt and sim_time from the module file. 
    Must be written as dt = X.Y and sim_time = A.B in the module file.
    The argument file_path is the name of the simulation module file.
    """
    
    search_dt = "dt = (\d+\.\d+)"
    search_sim_time = "sim_time = (\d+\.\d+)"
    with open(file_path, "r+") as file:
        file_contents = file.read()
        dt = re.findall(search_dt, file_contents)
        sim_time = re.findall(search_sim_time, file_contents)
        if len(dt) == 0:
            dt_result = float(dt_default)
        else:
            dt_result = float(dt[0])
        if len(sim_time) == 0:
            sim_time_result = sim_time_default
        else:
            sim_time_result = float(sim_time[0])

    return dt_result, sim_time_result
    
def loguniform_by_quantiles(low, high, size, num) -> list:
    quantiles = np.linspace(0, 1, num+2)[1:-1]
    power_samples = np.random.uniform(low=low, high=high, size=size)
    samples = [10**s for s in power_samples]
    x_cuts = [np.quantile(samples, q) for q in quantiles]
    parameter_values = [np.round(x, 5) for x in x_cuts]
    return parameter_values

def main() -> None:

    gamma_values = [1.99779, 3.79126, 6.96415, 13.05505, 23.31662, 42.81914, 77.91484, 149.97132, 275.89982, 520.30146]
    reg_param_values = [1.47881, 2.17266, 3.37139, 5.05364, 7.75259, 12.11043, 18.13678, 28.24211, 43.40776, 64.54524]
    combinations = list(itertools.product(gamma_values, reg_param_values))

    j = int(sys.argv[1])
    num_combinations = len(gamma_values)*len(reg_param_values)

    if j >= num_combinations:
        i = j % num_combinations
    else:
        i = j

    gamma = combinations[i][0]
    reg_param = combinations[i][1]

    moduleName = 'D:/Projects/GNN Research/Code Files/my_code_files/general/simulation_module.py'

    ##sys.stdout = open(os.devnull, 'w')  # Disable printing from simulations
    start_time_each_simulation = time.time()

    simulate(moduleName, reg_param=reg_param, gamma=gamma, iter_index=j, max_cells = max_cells_default)

    end_time_each_simulation = time.time()

    if end_time_each_simulation - start_time_each_simulation >= 1500:

        print("simulation time exceeding 1500 sec, removing the current simulation and skipping to the next...")

        ##dt_now = str(datetime.now())[:10]  ## get today's date
        dt_now = "2025-09-25"

        if j < 10:  ## generate folder's path
            output_dir = "D:/Projects/GNN Research/Data Files/_sim_pkl_data_gnn/"+dt_now+f"/iteration 00{j}/"+f"gamma={gamma}_reg_param={reg_param}_iter={j}/data"
        elif (j >= 10) and (j <100):
            output_dir = "D:/Projects/GNN Research/Data Files/_sim_pkl_data_gnn/"+dt_now+f"/iteration 0{j}/" + f"gamma={gamma}_reg_param={reg_param}_iter={j}/data"
        elif (j >= 100) and (j <1000):
            output_dir = "D:/Projects/GNN Research/Data Files/_sim_pkl_data_gnn/"+dt_now+f"/iteration {j}/" + f"gamma={gamma}_reg_param={reg_param}_iter={j}/data"
        else:
            print("We don't want the number of iterations greater than 999 since it's too computationally heavy")

        if os.path.exists(output_dir):
            os.remove(output_dir)
            print(f"File '{output_dir}' deleted successfully.")
        else:
            print(f"File '{output_dir}' does not exist.")

    print("Simulation completed")

if __name__ == "__main__": 
    main()
