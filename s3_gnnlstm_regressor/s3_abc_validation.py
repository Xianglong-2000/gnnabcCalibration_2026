import os
#os.environ['CM_NO_OPENCL'] = '1'  ### CPU only

import pickle
import sys
import subprocess
import string
import shutil
import uuid
import copy
import datetime
import math
import warnings

import pyabc
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import pickle as pk

import torch

import s3_models

np.seterr(all="ignore") #suppress warnings for cleaner output

import warnings
warnings.filterwarnings("ignore")

def ABCsimulation(params):
    return None

def model_runner(parameters:dict):

    # subprocess the iteration to genereate and save graph data
    path = "/work/x5bai/project/Code_Files/s3/s3_abc_iter.py"
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
    g_data_path = f"/work/x5bai/project/Data_Files/_simulator_graph_data_abc/{dt_sim}/abc_g_data_gamma={gamma}_reg_param={reg_param}/graphs.pkl"
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
    model = s3_models.GAT_Learner(in_channels=in_channels_model, 
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

    ##print("output: ",output)  ### check output structure
    #output_names = ["e1", "e2", "e3", "e4", "e5", "e6"]
    #result = {n:round(float(e),5) for n,e in zip(output_names,output.tolist()[0])}  ### in ln space
    ##print("result: ",result)

    p1 = float(output.tolist()[0][0])
    p2 = float(output.tolist()[0][1])
    p3 = float(output.tolist()[0][2])
    p4 = float(output.tolist()[0][3])
    p5 = float(output.tolist()[0][4])
    p6 = float(output.tolist()[0][5])
    p1 = (p1 - min_ss1)/(max_ss1 - min_ss1)  ### min-max normalization
    p2 = (p2 - min_ss2)/(max_ss2 - min_ss2)  ### min-max normalization
    p3 = (p3 - min_ss3)/(max_ss3 - min_ss3)  ### min-max normalization
    p4 = (p4 - min_ss4)/(max_ss4 - min_ss4)  ### min-max normalization
    p5 = (p5 - min_ss5)/(max_ss5 - min_ss5)  ### min-max normalization
    p6 = (p6 - min_ss6)/(max_ss6 - min_ss6)  ### min-max normalization
    result = {"e1": p1,  ### this result is in ln space
              "e2": p2,
              "e3": p3,
              "e4": p4,
              "e5": p5,
              "e6": p6,
              }  

    return result

if __name__ == "__main__": 

    # setup
    dt_sim = "2025-09-25"
    moduleName = "simulation_module.py"
    CellTypes = ['YFP']
    dt_default = 0.025
    max_cells = 140
    max_time = None
    in_channels_model = 31
    hidden_channels_model = 4*in_channels_model
    feature_embedding_size = 6

    # get ss min and max from 1000 simulations for GNN
    parent_dir = "/work/x5bai/project/Data_Files/_sim_mean_std_values_gnn/2025-09-25/"  
    final_path = parent_dir + "S3_test_1108_minmax.pkl" 
    #print(final_path) 
    with open(final_path, "rb") as file: 
        values = pk.load(file)  
    ##print(values)  
    min_ss1 = values["min ss1"] 
    min_ss2 = values["min ss2"] 
    min_ss3 = values["min ss3"] 
    min_ss4 = values["min ss4"] 
    min_ss5 = values["min ss5"] 
    min_ss6 = values["min ss6"] 
    max_ss1 = values["max ss1"] 
    max_ss2 = values["max ss2"] 
    max_ss3 = values["max ss3"] 
    max_ss4 = values["max ss4"] 
    max_ss5 = values["max ss5"] 
    max_ss6 = values["max ss6"] 
    print("min ss1: ", min_ss1)
    print("min ss2: ", min_ss2)
    print("min ss3: ", min_ss3)
    print("min ss4: ", min_ss4)
    print("min ss5: ", min_ss5)
    print("min ss6: ", min_ss6)
    print("max ss1: ", max_ss1)
    print("max ss2: ", max_ss2)
    print("max ss3: ", max_ss3)
    print("max ss4: ", max_ss4)
    print("max ss5: ", max_ss5)
    print("max ss6: ", max_ss6)

    saved_model_path = "/work/x5bai/project/Data_Files/_model_data_old/S3_test_1108.pth"

    # simple test
    #parameters = {'gamma': math.log(200), 'reg_param': math.log(35)}
    #summary_stat_dict = model_runner(parameters)
    #print("stop here for a test: ", summary_stat_dict)
    #breakpoint()

    # Step 0: load real-exp g_data
    g_data_path = "/work/x5bai/project/Data_Files/_exp_data/exp_graphs.pkl"
    with open(g_data_path, "rb") as file:
        data_tuple = pk.load(file)

    # Step 1: get sim-exp ss
    data_dict = {}
    for k, v in data_tuple.items():
        result = gnn_prediction(v)
        t = (v, result) 
        data_dict[k] = t

    e1_list = [v[-1]["e1"] for k, v in data_dict.items()]
    e2_list = [v[-1]["e2"] for k, v in data_dict.items()]
    e3_list = [v[-1]["e3"] for k, v in data_dict.items()]
    e4_list = [v[-1]["e4"] for k, v in data_dict.items()]
    e5_list = [v[-1]["e5"] for k, v in data_dict.items()]
    e6_list = [v[-1]["e6"] for k, v in data_dict.items()]

    exp_summary_stats = {}
    exp_summary_stats["e1"] = e1_list
    exp_summary_stats["e2"] = e2_list
    exp_summary_stats["e3"] = e3_list
    exp_summary_stats["e4"] = e4_list
    exp_summary_stats["e5"] = e5_list
    exp_summary_stats["e6"] = e6_list

    print("exp ss: ", exp_summary_stats)
    exp_ss_df = pd.DataFrame(exp_summary_stats)
    exp_ss_df.to_csv('/work/x5bai/project/Data_Files/_exp_data/exp_summary_stat_s3s4.csv', index=False)
    breakpoint()

    # Define settings
    n_sims = 24
    sample_uniform = False # Set to true to sample from uniform distribution
    gamma_lb = math.log(0.1)
    gamma_ub = math.log(1000)
    alpha_lb = math.log(0.01)
    alpha_ub = math.log(100)

    # Load database
    lower_bound = 0
    scale = 1
    prior = pyabc.Distribution(mu=pyabc.RV("uniform", lower_bound, scale))
    abc = pyabc.ABCSMC(ABCsimulation, prior)
    db_path = ("sqlite:////work/x5bai/project/Data_Files/_abc_results/" + "s3_expabc_re.db")
    run_id = 11
    history = abc.load(db_path, run_id)

    # Get probability density functions
    df, w = df, w = history.get_distribution(t=history.max_t)
    x_gamma, pdf_gamma = pyabc.visualization.kde.kde_1d(df, w, 'gamma', xmin=np.log(0.1), xmax=np.log(1000), numx=200, kde=None)
    x_alpha, pdf_alpha = pyabc.visualization.kde.kde_1d(df, w, 'reg_param', xmin=np.log(0.01), xmax=np.log(100), numx=200, kde=None)   
   
    sim_counter = 0  
    df = pd.DataFrame()
    for n in range(n_sims):
        # Sample from distribution
        if sample_uniform:
            print("Note: Sampling from uniform distribution")
            gamma = np.random.uniform(low=gamma_lb, high=gamma_ub)
            reg_param = np.random.uniform(low=alpha_lb, high=alpha_ub)
        else:
            gamma = random.choices(x_gamma, weights=pdf_gamma)[0]
            reg_param = random.choices(x_alpha, weights=pdf_alpha)[0]
        
        parameters = {'gamma': gamma, 'reg_param': reg_param}

        # Run model
        print(f'Running with parameters in ln space {parameters}')
        
        try:
            summary_stat_dict = model_runner(parameters)
        except:
            print(f"Run failed")
            summary_stat_dict = {'e1': 0, 
                                 'e2': 0, 
                                 'e3': 0, 
                                 'e4': 0,
                                 'e5': 0,
                                 'e6': 0}
               
        # Terminal output
        sim_counter += 1
        print(f'{sim_counter}/{n_sims} completed')
        
        new_row = pd.DataFrame(summary_stat_dict, index=[0])
        new_row = pd.concat([pd.DataFrame(parameters, index=[0]), new_row], axis=1)
        df = pd.concat([df, new_row], ignore_index=True)
    
    # Export data
    if sample_uniform:
        df.to_csv('/work/x5bai/project/Data_Files/_sampling/s3_expabc_re_ss_pri.csv', index=False)
    else:
        df.to_csv('/work/x5bai/project/Data_Files/_sampling/s3_expabc_re_ss_pos.csv', index=False)
    
    
