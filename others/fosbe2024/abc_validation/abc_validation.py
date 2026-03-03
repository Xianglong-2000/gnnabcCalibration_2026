import os
import pickle
import sys
import subprocess
import string
import shutil
import uuid
import copy
import datetime
import warnings

import pyabc
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
from CellModeller.Simulator import Simulator

import matplotlib.pyplot as plt

# Local/custom modules
from Scripts.run_ss_on_exp_sim.scripts.summary_statistics.AspectRatio import calc_aspect_ratio
from Scripts.run_ss_on_exp_sim.scripts.summary_statistics.Anisotropy import get_global_order_parameter
from Scripts.run_ss_on_exp_sim.scripts.summary_statistics.density_calculation import get_density_parameter
from Scripts.run_ss_on_exp_sim.scripts.summary_statistics.growth_rate_exp_deviation import get_exp_deviation, get_norm_growth_rate
from Scripts.run_ss_on_exp_sim.scripts.summary_statistics.convexity import cal_convexity
from Scripts.run_ss_on_exp_sim.scripts.helper_functions import helperFunctions 

'''
Instructions:
0. Setup the simulation module accordingly; ensure there is a time step variable dt = x.y
1. Define parameters, n_distributions, and replicates_list in main()
2. Set run_sims variable to False to skip running simulations
3. Run.
4. Determine the minimum number of replicates needed to get an A-measure below 0.56.
'''
np.seterr(all="ignore") #suppress warnings for cleaner output
warnings.filterwarnings("ignore") # ignore all warnings

cellmodeller_module = "simulation_module.py"
max_cells = 200

def ABCsimulation(params):
    return None

def run_model(parameters):
    # Run CellModeller simulation
    max_attempts = 3
    for attempt in range(max_attempts):
        # Defining input/output locations
        export_path = "data/" + str(uuid.uuid4())
        sys.stdout = open(os.devnull, 'w') # Disable printing from simulations
        # Run CellModeller simulation
        simulate(cellmodeller_module, parameters, export_path, max_cells=max_cells)
        sys.stdout = sys.__stdout__ # Re-enable printing
        print(f'Simulation completed with parameters: {parameters} at {datetime.datetime.now()}')
        break
        '''
        try:    
            # Run CellModeller simulation
            simulate(cellmodeller_module, parameters, export_path, max_cells=max_cells, max_time=max_time)
            sys.stdout = sys.__stdout__ # Re-enable printing
            print(f'Simulation completed with parameters: {parameters} at {datetime.datetime.now()}')
            break
        except:
            print(f'Simulation failed for parameters: {parameters}')
            #shututil.rmtree(export_path)      
        '''
        
    # Load cellStates
    pickle_list = helperFunctions.create_pickle_list(export_path)
    cells = helperFunctions.load_cellStates(export_path, pickle_list[-1]) # load last pickle file
    
    # Extract time parameters from simulation (could hard-code to avoid unnecessary repetition)
    sim_file = os.path.abspath(cellmodeller_module)
    dt = sim_time_parameters(sim_file)
    
    # Calculate summary statistics
    summary_stats = {}
    summary_stats['R2_exp_fit'] = get_exp_deviation(export_path, dt)
    summary_stats['density_parameter'] = get_density_parameter(cells)
    summary_stats['order_parameter'] = get_global_order_parameter(cells)
    summary_stats['aspect_ratio'] = calc_aspect_ratio(cells)
    summary_stats['convexity'] = cal_convexity(cells)
    summary_stats['growth_rate'] = get_norm_growth_rate(export_path, dt)

    # Write summary stats to file for convenience (optional) 
    file = open(export_path + "/summary_stats.txt","w")
    for key, value in summary_stats.items(): 
        file.write('%s: %.3f\n' % (key, value)) 
    file.close()
    
    return summary_stats

def simulate(modfilename, params, export_path, max_cells=None, max_time=None):
    """
    Same as batch.py with modificationsto allow external parameters and termination based on stepNum
    
    @param modfilename  module name (e.g., "Tutorial1.py")
    @param export_path  output directory name
    @param params       dictionary of parameter values passed to the simulation
    @param max_cells    number of cells where simulation terminates (int)
    @param max_time     simulation time where simulation terminates (float)
    """
    # Get module file name and append module to system path
    (path,name) = os.path.split(modfilename)
    modname = str(name).split('.')[0]
    sys.path.append(path)
    
    # Extract time parameters from simulation
    sim_file = os.path.abspath(modfilename)
    dt = sim_time_parameters(sim_file)
    
    # Create Simulator object
    sim = Simulator(modname, dt, saveOutput=True, outputDirName=export_path, psweep=True, params=params)
    
    # Run simulation and terminate when simulation reaches a certain number of cells
    if max_cells:
        while len(sim.cellStates) <= max_cells:
            sim.step()
        
    # Export pickle at final timestep
    sim.writePickle()
    
    # Delete sim object to prevent OOMing?? I guess it works, but I got cell index error!?
    del sim
    
def sim_time_parameters(file_path):
    """
    Extract the simulation time step from the module file. 
    Must be written as dt = X.Y in the module file
    
    @param  file_path   str of file name containing parameters to be modified
    @return dt          Simulation time step specified in CellModeller module file
    """
    search_dt = f'dt = (\d+\.\d+)'
    with open(file_path, "r+") as file:
        file_contents = file.read()
        dt = re.findall(search_dt, file_contents)
    return float(dt[0])
    
def write_to_pickle(data, file_name):
    # Open a file, where you want to store the data
    file = open(file_name, 'wb')

    # Dump information to that file
    pickle.dump(data, file)

    # Close the file
    file.close()
       
def main():
    # Define settings
    n_sims = 2
    sample_uniform = True # Set to true to sample from uniform distribution
    gamma_lb = np.log(0.1)
    gamma_ub = np.log(1000)
    alpha_lb = np.log(0.01)
    alpha_ub = np.log(100)

    # Load database
    lower_bound = 0
    scale = 1
    prior = pyabc.Distribution(mu=pyabc.RV("uniform", lower_bound, scale))
    abc = pyabc.ABCSMC(ABCsimulation, prior)
    db_path = ("sqlite:///" + "results.db")
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
            gamma = random.choices(x_gamma, weights=pdf_gamma)
            reg_param = random.choices(x_alpha, weights=pdf_alpha)
        
        parameters = {'gamma': gamma, 'reg_param': reg_param}
        # Run model
        print(f'Running with parameters {parameters}')
        
        try:
            summary_stat_dict = run_model(parameters)
        except:
            print(f"Run failed")
            summary_stat_dict = {'R2_exp_fit': 0, 
                                 'density_parameter': 0, 
                                 'order_parameter': 0, 
                                 'aspect_ratio': 0,
                                 'convexity': 0}
               
        # Terminal output
        sim_counter += 1
        print(f'{sim_counter}/{n_sims} completed')
        
        new_row = pd.DataFrame(summary_stat_dict, index=[0])
        new_row = pd.concat([pd.DataFrame(parameters, index=[0]), new_row], axis=1)
        df = pd.concat([df, new_row], ignore_index=True)
    
    # Export data
    if sample_uniform:
        df.to_csv('summary_stats_uniform.csv', index=False)
    else:
        df.to_csv('summary_stats_calibrated.csv', index=False)
                    
if __name__ == "__main__": 
    main()
