import os
os.environ['CM_NO_OPENCL'] = '1'  ### CPU only

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

sys.path.append('/work/x5bai/project/Code_Files/s5/')
import s5_expabc_runner_re

np.seterr(all="ignore") #suppress warnings for cleaner output

import warnings
warnings.filterwarnings("ignore")

def ABCsimulation(params):
    return None

def model_runner(parameters:dict):

    # subprocess the iteration to genereate and save graph data
    path = "/work/x5bai/project/Code_Files/s5/s5_abc_iter_re.py"
    envr = sys.executable
    time_limit = 3000

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

if __name__ == "__main__": 

    # CM setup
    dt_sim = "2025-13-01"
    moduleName = "simulation_module.py"
    CellTypes = ['YFP']
    dt_default = 0.025
    max_cells = 140
    max_time = None

    # simple test
    #parameters = {'gamma': math.log(200), 'reg_param': math.log(35)}
    #summary_stat_dict = model_runner(parameters)
    #print("stop here for a test: ", summary_stat_dict)
    #breakpoint()

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
    db_path = ("sqlite:////work/x5bai/project/Data_Files/_abc_results/" + "s5_expabc_kl.db")
    run_id = 1
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
            summary_stat_dict = {'Aspect ratio': 0, 
                                 'Order parameter': 0, 
                                 'Convexity': 0, 
                                 'Density': 0,
                                 'Agreement with exponential growth': 0,
                                 'Normalized growth rate': 0}
               
        # Terminal output
        sim_counter += 1
        print(f'{sim_counter}/{n_sims} completed')
        
        new_row = pd.DataFrame(summary_stat_dict, index=[0])
        new_row = pd.concat([pd.DataFrame(parameters, index=[0]), new_row], axis=1)
        df = pd.concat([df, new_row], ignore_index=True)
    
    # Export data
    if sample_uniform:
        df.to_csv('/work/x5bai/project/Data_Files/_sampling/s5_expabc_kl_ss_pri.csv', index=False)
    else:
        df.to_csv('/work/x5bai/project/Data_Files/_sampling/s5_expabc_kl_ss_pos.csv', index=False)
    
    
