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

sys.path.append('D:/Projects/GNN Research/Code Files/my_code_files/s1')
import models_gatregressor


def model_test(parameters:dict):

    # subprocess the iteration to genereate and save graph data
    path = "D:/Projects/GNN Research/Code Files/my_code_files/s1/abcsmc_iter_gatregressor.py"
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
    g_data_path = f"D:/Projects/GNN Research/Data Files/_simulator_graph_data_abc/2025-08-20/abc_g_data_gamma={gamma}_reg_param={reg_param}.pkl"
    print(g_data_path)
    with open(g_data_path, "rb") as file:
        data = pk.load(file)

    # gnn prediction as the ss
    result = gnn_prediction(data)
    ##print("parameter prediction: ", {k:round(10**v, 5) for k,v in result.items()})
    ##print("log10 parameter prediction: ", result)

    return result

def gnn_prediction(input_data) -> dict:
    """
    input = Data(x=[1681, 22], contact_edge_index=[2, 8346], lineage_edge_index=[2, 1440])
    output = {"gamma":100, "alpha":100}
    """

    # load the trained model
    model = models_gatregressor.GAT_Regressor(in_channels=in_channels_model, hidden_channels=hidden_channels_model,
                out_channels=feature_embedding_size)
    model.load_state_dict(torch.load(saved_model_path))
    model.eval()

    # predict gamma and reg_param
    with torch.no_grad():
        output = model(input_data)
    ##print(output)
    p1 = round(float(output.tolist()[0][0]*gamma_std+gamma_mean),5)  ### de-normalization
    p2 = round(float(output.tolist()[0][1]*reg_param_std+reg_param_mean),5)  ### de-normalization
    p1 = (p1-gamma_min)/(gamma_max-gamma_min)  ### min-max normalization
    p2 = (p2-reg_param_min)/(reg_param_max-reg_param_min)  ### min-max normalization
    result = {"gamma": p1, "alpha": p2}  ### this result is in log10 space

    return result
   
def distance_calculation(sim_stats, exp_stats):
    distance_list = []
    for x in sim_stats.keys():
        std_exp_stats = np.where(exp_stats[x] == 0, 1e-8, exp_stats[x])
        diff = np.abs(sim_stats[x] - exp_stats[x])/std_exp_stats
        distance_list.append(diff)
    distance = np.linalg.norm(distance_list)
    return distance 


if __name__ == '__main__':

    # set up local arguments to main() function
    date_simulations = "2025-08-20"
    moduleName = "simulation_module.py"
    in_channels_model = 31
    hidden_channels_model = 2*in_channels_model
    feature_embedding_size = 2
    CellTypes = ['YFP']
    dt_default = 0.025
    max_cells = 140
    max_time = None
    cell_type_mapping = {'YFP': 0}
    assign_cell_type = True
    use_grandmother_as_parent = False
    find_neighbors = True

    # get stacked mean and std from 1000 simulations for GNN
    parent_dir = "D:/Projects/GNN Research/Data Files/_sim_mean_std_values_gnn/2025-08-20/"  
    final_path = parent_dir + "CVIS_S1_minmaxtest.pkl"  
    print(final_path) 
    with open(final_path, "rb") as file: 
        values = pk.load(file)  
    ##print(values)  
    gamma_mean = values["gamma mean"] 
    gamma_std = values["gamma std"] 
    reg_param_mean = values["reg_param mean"] 
    reg_param_std = values["reg_param std"]  
    mean_dict = values["feature mean"] 
    std_dict = values["feature std"] 
    gamma_min = values["gamma min"] 
    gamma_max = values["gamma max"] 
    reg_param_min = values["reg_param min"] 
    reg_param_max = values["reg_param max"]  

    print("log(gamma) mean: ", gamma_mean)
    print("log(gamma) std: ", gamma_std)
    print("log(reg_param) mean: ", reg_param_mean)
    print("log(reg_param) std: ", reg_param_std)
    ##print("feature mean: ", mean_dict)
    ##print("feature std: ", std_dict)
    print("log(gamma) min: ", gamma_min)
    print("log(gamma) max: ", gamma_max)
    print("log(reg_param) min: ", reg_param_min)
    print("log(reg_param) max: ", reg_param_max)

    saved_model_path = "D:/Projects/GNN Research/Data Files/_model_data_new/CVIS_S1.pth"

    # 31 features
    include_columns = ['ImageNumber', #
                       'ObjectNumber', #
                       'id', #
                       'parent_id', #
                       'AreaShape_Area', #
                       'AreaShape_MajorAxisLength', #
                       'AreaShape_Orientation', #
                       'cellAge', #
                       'LifeHistory', #
                       'TrajectoryX', #
                       'TrajectoryY', #
                       'Direction_of_Motion', #
                       'Motion_Alignment_Angle', #
                       'Source_Neighbor_Avg_TrajectoryX', #
                       'Source_Neighbor_Avg_TrajectoryY', #
                       'Division_TimeStep', #
                       'Daughter_Mother_Length_Ratio', #
                       'Total_Daughter_Mother_Length_Ratio', #
                       'Max_Daughter_Mother_Length_Ratio', #
                       'Daughter_Avg_TrajectoryX', #
                       'Daughter_Avg_TrajectoryY', #
                       'Neighbor_Difference_Count', #
                       'Neighbor_Shared_Count', #
                       'Average_Length', #
                       'Velocity', #
                       'Instant_Velocity', #
                       'Average_Instant_Velocity', #
                       'Elongation_Rate', #
                       'Instant_Elongation_Rate', #
                       'startVol', #
                       'targetVol', #
                       'Prev_MajorAxisLength', #
                       'Bacterium_Movement', #
                       'dir_1', 'dir_2', #
                       ]
 
    ##parameters = {"gamma":math.log10(250), "reg_param":math.log10(30)}  ### for a simple test only
    ##model(parameters)  ### for a simple test only
    ##print()  ### for a simple test only

    # Step 0: load sim-exp g_data
    g_data_path = "D:/Projects/GNN Research/Data Files/_sim_graph_data_gnn/2025-08-24/CVIS_S1_simexp.pkl"
    with open(g_data_path, "rb") as file:
        data_tuple = pk.load(file)

    # Step 1: get sim-exp ss
    data_dict = {}
    for k, v in data_tuple.items():
        result = gnn_prediction(v)
        t = (v, result) 
        data_dict[k] = t

    gamma_list = [v[-1]["gamma"] for k, v in data_dict.items()]
    alpha_list = [v[-1]["alpha"] for k, v in data_dict.items()]
    exp_summary_stats = {}
    exp_summary_stats["gamma"] = statistics.mean(gamma_list)
    exp_summary_stats["alpha"] = statistics.mean(alpha_list)

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
    ##n_cores = 4
    population_size = 100
    min_epsilon = 0.07071
    max_populations = 100 

    # Step x: create abc model
    abc = pyabc.ABCSMC(model_test, 
                       prior, 
                       distance_calculation, 
                       population_size=population_size, 
                       sampler=pyabc.sampler.SingleCoreSampler()    ### will cause error if set it to multiplecoresampler for some reasons
                       )

    # Step x: create database      
    """             
    db_path = "CVIS_S1_minmaxtest.db"
    history = abc.new("sqlite:///D:/Projects/GNN Research/Data Files/abc_results/" + db_path, exp_summary_stats)
    print("ABC-SMC run ID:", history.id)
    """

    # to resume a stored run only
    #"""
    db_path = "CVIS_S1_minmaxtest.db"
    history = abc.load("sqlite:///D:/Projects/GNN Research/Data Files/abc_results/" + db_path, 1)
    print("num of completed populations: ", history.n_populations)
    print("ABC-SMC run ID: ", history.id)
    ##breakpoint()
    #"""
    
    # Step x: run
    start_time_all = time.time()
    history = abc.run(minimum_epsilon=min_epsilon, 
                      max_nr_populations=max_populations) 

    end_time_all = time.time()  ### about 20 hours for population_size*num_populations = 500
    print(f"Training time: {(end_time_all - start_time_all)/3600:.2f} hours")