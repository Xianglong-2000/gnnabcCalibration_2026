import os
import sys
sys.path.append('/work/x5bai/project/Code_Files/CellModeller-ingallslab')
import re
import itertools
import numpy as np
from datetime import datetime
import psutil
from CellModeller.Simulator import Simulator
import random
import time
import shutil

import warnings
warnings.filterwarnings("ignore")

# setup for defaults
dt_default = 0.025
sim_time_default = None
max_cells_default = 140
dt_now = "2026-02-06"

def simulate(modfilename, reg_param, gamma, iter_index = 0, max_cells=None, max_time=None):
# simulate(modfilename, params, export_path, max_cells=None, max_time=None):  ### Aaron

    (path,name) = os.path.split(modfilename)
    modname = str(name).split('.')[0]
    sys.path.append(path)

    if iter_index < 10:  ## generate folder's path
        output_dir = "/work/x5bai/project/Data_Files/_sim_pkl_data_gnn/"+dt_now+f"/iteration 00{iter_index}/"+f"gamma={gamma}_reg_param={reg_param}_iter={iter_index}/data"
    elif (iter_index >= 10) and (iter_index <100):
        output_dir = "/work/x5bai/project/Data_Files/_sim_pkl_data_gnn/"+dt_now+f"/iteration 0{iter_index}/" + f"gamma={gamma}_reg_param={reg_param}_iter={iter_index}/data"
    elif (iter_index >= 100) and (iter_index <1000):
        output_dir = "/work/x5bai/project/Data_Files/_sim_pkl_data_gnn/"+dt_now+f"/iteration {iter_index}/" + f"gamma={gamma}_reg_param={reg_param}_iter={iter_index}/data"
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
    # Simulator(modname, dt, saveOutput=True, outputDirName=export_path, psweep=True, params=params)  ### Aaron

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

def main() -> None:

    i = int(sys.argv[1])
    gamma = round(float(sys.argv[2]),5)
    reg_param = round(float(sys.argv[3]),5)

    moduleName = '/work/x5bai/project/Code_Files/general/simulation_module.py'

    sys.stdout = open(os.devnull, 'w')  # Disable printing from simulations

    start_time_each_simulation = time.time()

    try:
        simulate(moduleName, reg_param=reg_param, gamma=gamma, iter_index=i, max_cells = max_cells_default)
    except Exception as e:
        output_dir = f"/work/x5bai/project/Data_Files/_errors/CM/{dt_now}/gamma={gamma}_reg_param={reg_param}"
        os.makedirs(output_dir, exist_ok=True)

        pkl_path = f"/work/x5bai/project/Data_Files/_simulator_pkl_data_abc/{dt_now}/gamma={gamma}_reg_param={reg_param}"
        shutil.rmtree(pkl_path, ignore_errors=True)

        return main()

    end_time_each_simulation = time.time()

    if end_time_each_simulation - start_time_each_simulation >= 1500:

        # delete current sim
        if i < 10:  ## generate folder's path
            output_dir = "/work/x5bai/project/Data_Files/_sim_pkl_data_gnn/"+dt_now+f"/iteration 00{i}/"+f"gamma={gamma}_reg_param={reg_param}_iter={i}/data"
        elif (i >= 10) and (i <100):
            output_dir = "/work/x5bai/project/Data_Files/_sim_pkl_data_gnn/"+dt_now+f"/iteration 0{i}/" + f"gamma={gamma}_reg_param={reg_param}_iter={i}/data"
        elif (i >= 100) and (i <1000):
            output_dir = "/work/x5bai/project/Data_Files/_sim_pkl_data_gnn/"+dt_now+f"/iteration {i}/" + f"gamma={gamma}_reg_param={reg_param}_iter={i}/data"
        else:
            print("We don't want the number of iterations greater than 999 since it's too computationally heavy")
        shutil.rmtree(output_dir, ignore_errors=True)

        # create error file
        error_dir = f"/work/x5bai/project/Data_Files/_errors/CM/{dt_now}/gamma={gamma}_reg_param={reg_param}"
        os.makedirs(error_dir, exist_ok=True)

        # run the main() again
        return main()

if __name__ == "__main__": 
    main()
