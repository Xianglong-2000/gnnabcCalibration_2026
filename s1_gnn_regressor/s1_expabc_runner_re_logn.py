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

sys.path.append('/work/x5bai/project/Code_Files/s1/')
import s1_models


def model_runner(parameters:dict):

    # subprocess the iteration to genereate and save graph data
    path = "/work/x5bai/project/Code_Files/s1/s1_abc_iter.py"
    envr = sys.executable
    #time_limit = 3000

    subprocess.run([envr, path, str(parameters["gamma"]), str(parameters["reg_param"])],
                    capture_output=True, 
                    text=True,
                    #timeout = time_limit, 
                    )

    # load the saved graph data
    gamma = round(math.exp(parameters["gamma"]), 5)
    reg_param = round(math.exp(parameters["reg_param"]), 5)
    g_data_path = f"/work/x5bai/project/Data_Files/_simulator_graph_data_abc/{date_simulations}/abc_g_data_gamma={gamma}_reg_param={reg_param}/graphs.pkl"
    print(g_data_path)
    with open(g_data_path, "rb") as file:
        data = pk.load(file)

    # gnn prediction as the ss
    result = gnn_prediction(data)
    print("simulator ss: ", result)

    return result

def gnn_prediction(input_data) -> dict:
    """
    input = Data(x=[1681, 22], contact_edge_index=[2, 8346], lineage_edge_index=[2, 1440])
    output = {"gamma":100, "alpha":100}
    """

    # load the trained model
    model = s1_models.GAT_Regressor(in_channels=in_channels_model,
                                    hidden_channels=hidden_channels_model,
                                    out_channels=feature_embedding_size)
    state = torch.load(saved_model_path, map_location="cpu")  ## debugging multiprocessor issue
    model.load_state_dict(state)  ## debugging multiprocessor issue
    device = torch.device("cpu")  ## debugging multiprocessor issue
    model.to(device)  ## debugging multiprocessor issue
    input_data = input_data.to(device)  ## debugging multiprocessor issue

    # predict graph embeddings
    model.eval()
    with torch.no_grad():
        output = model(input_data)

    ##print(output)
    p1 = round(float(output.tolist()[0][0]*gamma_std+gamma_mean),5)  ### de-normalization
    p2 = round(float(output.tolist()[0][1]*reg_param_std+reg_param_mean),5)  ### de-normalization
    p1 = (p1-gamma_min)/(gamma_max-gamma_min)  ### min-max normalization
    p2 = (p2-reg_param_min)/(reg_param_max-reg_param_min)  ### min-max normalization
    result = {"gamma": p1, "alpha": p2}  ### this result is in ln space

    return result
   
def distance_calculation(sim_stats, exp_stats):
    distance_list = []
    for x in sim_stats.keys():
        diff = np.abs(sim_stats[x] - exp_stats[x])/exp_stats[x]
        distance_list.append(diff)
    distance = np.linalg.norm(distance_list)
    #print("distance_list: ", distance_list)
    #print("distance: ", distance)
    return distance 

if __name__ == '__main__':

    # set up local arguments to main() function
    date_simulations = "2025-11-11"
    moduleName = "simulation_module.py"
    in_channels_model = 31
    hidden_channels_model = 4*in_channels_model
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
    parent_dir = "/work/x5bai/project/Data_Files/_sim_mean_std_values_gnn/2025-11-11/"  
    final_path = parent_dir + "S1_test_1111.pkl"
    #print(final_path)  
    with open(final_path, "rb") as file: 
        values = pk.load(file)  
    ##print(values)  
    gamma_mean = values["ln(gamma mean)"] 
    gamma_std = values["ln(gamma std)"] 
    reg_param_mean = values["ln(reg_param mean)"] 
    reg_param_std = values["ln(reg_param std)"]  
    mean_dict = values["feature mean"] 
    std_dict = values["feature std"] 
    gamma_min = values["ln(gamma min)"] 
    gamma_max = values["ln(gamma max)"] 
    reg_param_min = values["ln(alpha min)"] 
    reg_param_max = values["ln(alpha max)"]  

    print("ln(gamma) mean: ", gamma_mean)
    print("ln(gamma) std: ", gamma_std)
    print("ln(reg_param) mean: ", reg_param_mean)
    print("ln(reg_param) std: ", reg_param_std)
    ##print("feature mean: ", mean_dict)
    ##print("feature std: ", std_dict)
    print("ln(gamma) min: ", gamma_min)
    print("ln(gamma) max: ", gamma_max)
    print("ln(reg_param) min: ", reg_param_min)
    print("ln(reg_param) max: ", reg_param_max)

    saved_model_path = "/work/x5bai/project/Data_Files/_model_data_new/S1_test_1111.pth"

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

    # Step 0: load real-exp g_data
    g_data_path = "/work/x5bai/project/Data_Files/_exp_data/exp_graphs.pkl"
    with open(g_data_path, "rb") as file:
        data_tuple = pk.load(file)

    # Step 1: get real-exp ss
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

    ##print("gamma list: ", gamma_list)
    ##print("alpha list: ", alpha_list)

    print("exp ss: ", exp_summary_stats)

    ##breakpoint()

    # Step 2: get prior
    param_config = {'gamma': [math.log(0.1), math.log(1000)], 'reg_param': [math.log(0.01), math.log(100)]}  ### range of prior of ln gamma and alpha
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
    min_epsilon = 0.12247
    max_populations = 10

    # Step x: create database and abc model
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
    #"""
    db_path = "S1_test_1111.db"
    history = abc.new("sqlite:////work/x5bai/project/Data_Files/_abc_results/" + db_path, exp_summary_stats)
    print("ABC-SMC run ID:", history.id)
    #"""

    # to resume a stored run only
    """
    db_path = "S1_test_1115.db"
    history = abc.load("sqlite:////work/x5bai/project/Data_Files/_abc_results/" + db_path, 1)  ## pick up where we were stopped, need to check id using jupyter
    print("num of completed populations: ", history.n_populations)
    print("ABC-SMC run ID: ", history.id)
    """
    
    # Step x: run
    history = abc.run(minimum_epsilon=min_epsilon, 
                      max_nr_populations=max_populations) 

    end_time_all = time.time()  ### about 20 hours for population_size*num_populations = 500
    print(f"Training time: {(end_time_all - start_time_all)/3600:.2f} hours")