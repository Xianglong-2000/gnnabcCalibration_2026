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

sys.path.append('/work/x5bai/project/Code_Files/CellModeller-ingallslab')
from Scripts.run_ss_on_exp_sim.scripts.helper_functions import helperFunctions
from Scripts.run_ss_on_exp_sim.scripts.summary_statistics.AspectRatio import calc_aspect_ratio
from Scripts.run_ss_on_exp_sim.scripts.summary_statistics.Anisotropy import get_global_order_parameter
from Scripts.run_ss_on_exp_sim.scripts.summary_statistics.density_calculation import get_density_parameter
from Scripts.run_ss_on_exp_sim.scripts.summary_statistics.growth_rate_exp_deviation import get_exp_deviation, get_norm_growth_rate
from Scripts.run_ss_on_exp_sim.scripts.summary_statistics.convexity_smart import cal_convexity

def model_runner(parameters:dict):

    # subprocess the iteration to genereate and save graph data
    path = "/work/x5bai/project/Code_Files/s5/s5_abc_iter_re.py"
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

    # load the saved pickle data
    gamma = round(math.exp(parameters["gamma"]), 5)
    reg_param = round(math.exp(parameters["reg_param"]), 5)
    folder_path = f"/work/x5bai/project/Data_Files/_simulator_pkl_data_abc/{dt_sim}/gamma={gamma}_reg_param={reg_param}"
    pkl_path = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))][0] + "/summary_stats.pkl"
    #print(pkl_path)

    with open(pkl_path, "rb") as file:
        result = pk.load(file)
    
    #print(result)

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

    # Step 0: real-exp data ss
    ss_df = pd.read_csv("/work/x5bai/project/Code_Files/s5/exp_summary_stat.csv")
    Aspect_ratio_list = ss_df["Aspect ratio"]
    Convexity_list = ss_df["Convexity"]
    Order_parameter_list = ss_df["Order parameter"]
    Density_list = ss_df["Density"]
    Agreement_with_exponential_growth_list = ss_df["Agreement with exponential growth"]
    Normalized_growth_rate_list = ss_df["Normalized growth rate"]

    exp_summary_stats = {}
    exp_summary_stats["Aspect ratio"] = np.mean(Aspect_ratio_list)
    exp_summary_stats['Convexity'] = np.mean(Convexity_list)
    exp_summary_stats["Order parameter"] = np.mean(Order_parameter_list)
    exp_summary_stats["Density"] = np.mean(Density_list)
    exp_summary_stats["Agreement with exponential growth"] = np.mean(Agreement_with_exponential_growth_list)
    exp_summary_stats["Normalized growth rate"] = np.mean(Normalized_growth_rate_list)
    print("exp ss: ", exp_summary_stats)

    # Step 2: get prior
    param_config = {'gamma': [math.log(0.1), math.log(1000)], 'reg_param': [math.log(0.01), math.log(100)]}  ### range of prior of log10 gamma and alpha
    prior_distributions = {}
    for parameter_name in param_config.keys():
        param_low = param_config[parameter_name][0]
        param_hi = param_config[parameter_name][1]
        width = abs(param_hi - param_low)
        prior_distributions[parameter_name] = {"type": "uniform", "args": (param_low, width), "kwargs": {}}
    prior = pyabc.Distribution.from_dictionary_of_dictionaries(prior_distributions)

    # Define ABC-SMC settings
    n_cores = 10
    population_size = 100  
    min_epsilon = 0.05  
    max_populations = 20

    # Step x: create abc model
    abc = pyabc.ABCSMC(model_runner, 
                       prior, 
                       distance_function=distance_calculation, 
                       population_size=population_size,
                       sampler=pyabc.sampler.MulticoreParticleParallelSampler(n_procs=n_cores),
                       #sampler=pyabc.sampler.MulticoreEvalParallelSampler(n_procs=n_cores),
                       #sampler=pyabc.sampler.SingleCoreSampler(),
                       #eps=pyabc.QuantileEpsilon(initial_epsilon=6.79301462, alpha=0.5, weighted=False)
                       #eps=pyabc.QuantileEpsilon(alpha=0.5, weighted=False)
                       )

    # Step x: create database                   
    db_path = "s5_test01.db"
    history = abc.new("sqlite:///" + "/work/x5bai/project/Code_Files/s5/" + db_path, exp_summary_stats)
    print("ABC-SMC run ID:", history.id)

    # to resume a stored run only
    """
    db_path = "CVIS_S5_EXP.db"
    history = abc.load("sqlite:///D:/Projects/GNN Research/Data Files/abc_results/" + db_path, 1)  ## pick up where we were stopped, need to check id using jupyter
    print("num of completed populations: ", history.n_populations)
    print("ABC-SMC run ID: ", history.id)
    """
    
    # Step x: run
    history = abc.run(minimum_epsilon=min_epsilon, 
                      max_nr_populations=max_populations) 

    end_time_all = time.time()  ### about 20 hours for population_size*num_populations = 500
    print(f"Training time: {(end_time_all - start_time_all)/3600:.2f} hours")