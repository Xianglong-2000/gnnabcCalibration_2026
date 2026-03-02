import numpy as np
import pandas as pd
import pyabc
import torch
from torch_geometric.data import Data
import re
import math
from CellModeller.Simulator import Simulator
import statistics
import networkx as nx
import os
import sys
import pickle as pk
from pathlib import Path
import subprocess
import shutil
import time
import warnings
warnings.filterwarnings("ignore")

sys.path.append('/work/x5bai/project/Code_Files/general')  # for simulation_module.py

sys.path.append('/work/x5bai/project/Code_Files/CellModeller-ingallslab')
from Scripts.run_ss_on_exp_sim.scripts.helper_functions import helperFunctions
from Scripts.run_ss_on_exp_sim.scripts.summary_statistics.AspectRatio import calc_aspect_ratio
from Scripts.run_ss_on_exp_sim.scripts.summary_statistics.Anisotropy import get_global_order_parameter
from Scripts.run_ss_on_exp_sim.scripts.summary_statistics.density_calculation import get_density_parameter
from Scripts.run_ss_on_exp_sim.scripts.summary_statistics.growth_rate_exp_deviation import get_exp_deviation, get_norm_growth_rate
from Scripts.run_ss_on_exp_sim.scripts.summary_statistics.convexity_smart import cal_convexity

def model_iter(parameters:dict):
    """
    input: log10 gamma and log10 alpha
    go through things in order: simulation -> read last pickle file -> calculate summary statistics
    """

    # set up parameters back to regular space for simulations
    gamma = round(math.exp(parameters["gamma"]), 5)
    reg_param = round(math.exp(parameters["reg_param"]), 5)
    print(f"params for simulation in regular space: gamma = {gamma}, alpha = {reg_param}")

    # defining input/output locations
    export_path = f"/work/x5bai/project/Data_Files/_simulator_pkl_data_abc/{date_simulations}/gamma={gamma}_reg_param={reg_param}"
    os.makedirs(export_path, exist_ok=True)
    sys.stdout = open(os.devnull, 'w')  # Disable printing from simulations

    # get pickles
    start_time_each_simulation = time.time()
    simulate(moduleName, reg_param=reg_param, gamma=gamma, max_cells = max_cells, export_path = export_path)
    end_time_each_simulation = time.time()
    if end_time_each_simulation - start_time_each_simulation >= 1200:
        if os.path.exists(export_path):
            os.remove(export_path)
            print(f"File '{export_path}' deleted successfully.")
        else:
            print(f"File '{export_path}' does not exist.")
    sys.stdout = sys.__stdout__  # Re-enable printing
    ##print("Simulation completed")

    # Load cellStates
    pickle_path = [os.path.join(export_path, f) for f in os.listdir(export_path) if os.path.isdir(os.path.join(export_path, f))][0]
    ##print(pickle_path)  ### check
    pickle_list = helperFunctions.create_pickle_list_full_path(pickle_path)
    ##print(pickle_list[-1])  ### check
    cells = helperFunctions.load_cellStates_full_path(pickle_list[-1])  # load last pickle file

    # Extract time parameters from simulation (could hard-code to avoid unnecessary repetition)
    sim_file = "/work/x5bai/project/Code_Files/general/"+moduleName
    dt, _ = sim_time_parameters(sim_file)

    # Calculate summary statistics
    summary_stats = {}
    summary_stats["Aspect ratio"] = calc_aspect_ratio(cells)  ###
    summary_stats["Order parameter"] = get_global_order_parameter(cells)  ###
    summary_stats['Convexity'] = cal_convexity(cells)  ###
    summary_stats["Density"] = get_density_parameter(cells)  ###
    summary_stats["Agreement with exponential growth"] = get_exp_deviation(pickle_path, dt)  ###
    summary_stats["Normalized growth rate"] = get_norm_growth_rate(pickle_path, dt, n_max=225, t_min=72*3/60)  ###

    # save as pickle file
    with open(pickle_path + "/summary_stats.pkl", 'wb') as f:
        pk.dump(summary_stats, f)
    print("simulator ss: ", summary_stats)

    return summary_stats


def simulate(modfilename, reg_param, gamma, max_cells=None, max_time=None, export_path=None):

    (path,name) = os.path.split(modfilename)
    modname = str(name).split('.')[0]
    sys.path.append(path)

    output_dir = export_path  
    ##print(output_dir)

    os.makedirs(output_dir, exist_ok=True)  # create a new folder if it doesn't exist and do nothing if it exists
    os.chdir(output_dir)  # generate the simulated data in that folder

    # Extract dt and sim_time from the simulation module
    sim_file = "/work/x5bai/project/Code_Files/general/"+modfilename
    (dt, sim_time) = sim_time_parameters(sim_file)

    params = {"gamma": gamma, "reg_param": reg_param}
    #print("check parameters right before the current simulation: ", params.values())
    sim = Simulator(modname, dt, pickleSteps=2, saveOutput = True, psweep=True, params=params)

    if max_cells:
        while len(sim.cellStates) <= max_cells:
            sim.step()


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
            sim_time_result = max_time
        else:
            sim_time_result = float(sim_time[0])

    return dt_result, sim_time_result


if __name__ == '__main__':
    # set up local arguments to main() function
    date_simulations = "2025-13-01"  ### there's no training so just make up any date we want
    moduleName = "simulation_module.py"
    CellTypes = ['YFP']
    dt_default = 0.025
    max_cells = 140
    max_time = None

    ##gamma = math.log10(250)  ### for a simple test only
    ##reg_param = math.log10(35)  ### for a simple test only
    gamma = float(sys.argv[1])
    reg_param = float(sys.argv[2])

    parameters = {"gamma":gamma, "reg_param":reg_param}

    #print(parameters)
    model_iter(parameters)
    
