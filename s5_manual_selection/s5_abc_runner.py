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

sys.path.append('D:/Projects/Github Packages/CellModeller-ingallslab')
from Scripts.run_ss_on_exp_sim.scripts.helper_functions import helperFunctions
from Scripts.run_ss_on_exp_sim.scripts.summary_statistics.AspectRatio import calc_aspect_ratio
from Scripts.run_ss_on_exp_sim.scripts.summary_statistics.Anisotropy import get_global_order_parameter
from Scripts.run_ss_on_exp_sim.scripts.summary_statistics.density_calculation import get_density_parameter
from Scripts.run_ss_on_exp_sim.scripts.summary_statistics.growth_rate_exp_deviation import get_exp_deviation, get_norm_growth_rate
from Scripts.run_ss_on_exp_sim.scripts.summary_statistics.convexity_smart import cal_convexity

def model_test(parameters:dict):

    # subprocess the iteration to genereate and save graph data
    path = "D:/Projects/GNN Research/Code Files/my_code_files/s5/abcsmc_iter_traditionalss.py"
    envr = sys.executable
    time_limit = 3000
    try:
        subprocess.run([envr, path, str(parameters["gamma"]), str(parameters["reg_param"])],
                        check = True,
                        timeout = time_limit,  # Set timeout in seconds
                        )
    except subprocess.TimeoutExpired:
        print(f"Timeout after {time_limit}s — retrying...")
        subprocess.run([envr, path, str(parameters["gamma"]), str(parameters["reg_param"])],
                        check = True,
                        timeout = time_limit,  # Set timeout in seconds
                        )
    except subprocess.CalledProcessError as e:
        print(f"Script failed with error code {e.returncode}")
        subprocess.run([envr, path, str(parameters["gamma"]), str(parameters["reg_param"])],
                        check = True,
                        timeout = time_limit,  # Set timeout in seconds
                        )

    # load the saved graph data
    gamma = round(10**parameters["gamma"], 5)
    reg_param = round(10**parameters["reg_param"], 5)
    folder_path = f"D:/Projects/GNN Research/Data Files/_simulator_pkl_data_abc/{dt_sim}/gamma={gamma}_reg_param={reg_param}"
    pkl_path = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))][0] + "/summary_stats.pkl"
    print(pkl_path)

    with open(pkl_path, "rb") as file:
        result = pk.load(file)
    
    print(result)

    return result

   
def distance_calculation(sim_stats, exp_stats):
    distance_list = []
    for x in sim_stats.keys():
        diff = np.abs(sim_stats[x] - exp_stats[x])/exp_stats[x]
        distance_list.append(diff)
    distance = np.linalg.norm(distance_list)
    return distance 

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

    start_time_all = time.time()

    # set up local arguments to main() function
    dt_sim = "2025-13-01"
    moduleName = "simulation_module.py"
    CellTypes = ['YFP']
    dt_default = 0.025
    max_cells = 140
    max_time = None
    cell_type_mapping = {'YFP': 0}
    assign_cell_type = True
    use_grandmother_as_parent = False
    find_neighbors = True

    ##arameters = {"gamma":math.log10(250), "reg_param":math.log10(30)}  ### for a simple test only
    ##result = model_test(parameters)  ### for a simple test only
    ##print("+-+",result)  ### for a simple test only
    ##breakpoint()  ### for a simple test only

    # Step 0: load sim-exp pkl data
    folder_path = "D:/Projects/GNN Research/Data Files/_sim_pkl_data_gnn/2025-10-20"
    iterations = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
    ss_list_1 = []
    ss_list_2 = []
    ss_list_3 = []
    ss_list_4 = []
    ss_list_5 = []
    ss_list_6 = []
    for i in iterations:
        j = [os.path.join(i, f) for f in os.listdir(i) if os.path.isdir(os.path.join(i, f))][0] + "/data"
        k = [os.path.join(j, f) for f in os.listdir(j) if os.path.isdir(os.path.join(j, f))][0]
        print(k)  ### check
        pickle_list = helperFunctions.create_pickle_list_full_path(k)
        ##print(pickle_list[-1])  ### check
        cells = helperFunctions.load_cellStates_full_path(pickle_list[-1])
        sim_file = "D:/Projects/GNN Research/Code Files/my_code_files/general/"+moduleName
        dt,_ = sim_time_parameters(sim_file)
        ss_list_1.append(calc_aspect_ratio(cells))
        ss_list_2.append(get_global_order_parameter(cells))
        ss_list_3.append(cal_convexity(cells))
        ss_list_4.append(get_density_parameter(cells))
        ss_list_5.append(get_exp_deviation(k, dt))
        ss_list_6.append(get_norm_growth_rate(k, dt, n_max=225, t_min=72*3/60))
  
    # Calculate summary statistics
    exp_summary_stats = {}
    exp_summary_stats["Aspect ratio"] = statistics.mean(ss_list_1)
    exp_summary_stats["Order parameter"] = statistics.mean(ss_list_2)
    exp_summary_stats['Convexity'] = statistics.mean(ss_list_3)
    exp_summary_stats["Density"] = statistics.mean(ss_list_4)
    exp_summary_stats['Agreement with exponential growth'] = statistics.mean(ss_list_5)
    exp_summary_stats['Normalized growth rate'] = statistics.mean(ss_list_6)
    print("exp_summary_stats: ", exp_summary_stats)

    # Step 2: get prior
    param_config = {'gamma': [0, 3], 'reg_param': [0, 2]}  ### range of prior of log10 gamma and alpha
    prior_distributions = {}
    for parameter_name in param_config.keys():
        param_low = param_config[parameter_name][0]
        param_hi = param_config[parameter_name][1]
        width = abs(param_hi - param_low)
        prior_distributions[parameter_name] = {"type": "uniform", "args": (param_low, width), "kwargs": {}}
    prior = pyabc.Distribution.from_dictionary_of_dictionaries(prior_distributions)

    # Define ABC-SMC settings
    n_cores = 2
    population_size = 100  ### set it to 4 for a simple test
    min_epsilon = 0.1   ### set it to 0.5 for a simple test
    max_populations = 100  ### set it to 5 for a simple test

    # Step x: create abc model
    abc = pyabc.ABCSMC(model_test, 
                       prior, 
                       distance_calculation, 
                       population_size=population_size, 
                       sampler=pyabc.sampler.SingleCoreSampler()    ### will cause error if set it to multiplecoresampler for some reasons
                       )

    # Step x: create database                   
    db_path = "CVIS_S5.db"
    history = abc.new("sqlite:///D:/Projects/GNN Research/Data Files/abc_results/" + db_path, exp_summary_stats)
    print("ABC-SMC run ID:", history.id)

    # to resume a stored run only
    """
    db_path = "CVIS_S5.db"
    history = abc.load("sqlite:///D:/Projects/GNN Research/Data Files/abc_results/" + db_path, 1)  ## pick up where we were stopped, need to check id using jupyter
    print("num of completed populations: ", history.n_populations)
    print("ABC-SMC run ID: ", history.id)
    """
    
    # Step x: run
    history = abc.run(minimum_epsilon=min_epsilon, 
                      max_nr_populations=max_populations) 

    end_time_all = time.time()  ### about 20 hours for population_size*num_populations = 500
    print(f"Training time: {(end_time_all - start_time_all)/3600:.2f} hours")