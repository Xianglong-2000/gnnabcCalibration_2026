"""
This is the main script that runs pyabc and CellModeller. 

Instructions:
1. Set cellmodeller_module to the name of the simulation module (e.g., Tutorial1a.py)
2. In the main function, set max_cells to the number of cells where the simulation will end, or,
    set max_time to the simulation time where the simulation will end
3. Set the summary statistics to calculate in the model() function
4. Define the ABC calibration settings in the main function
5. Set experimental values for summary statistics in the main function;
    ensure the chosen summary statistics are the same as the simulated ones
6. Define prior distributions for each parameter in the main function
7. Run by entering the command: python pyabc_cellmodeller.py
"""

# Standard python modules
import multiprocessing
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import re
import sys
import uuid
import shutil
import warnings
import datetime
import gc

# Installed modules
from CellModeller.Simulator import Simulator
import pyabc
from pyabc.populationstrategy import AdaptivePopulationSize

from Scripts.run_ss_on_exp_sim.scripts.summary_statistics.AspectRatio import calc_aspect_ratio
from Scripts.run_ss_on_exp_sim.scripts.summary_statistics.Anisotropy import get_global_order_parameter
from Scripts.run_ss_on_exp_sim.scripts.summary_statistics.density_calculation import get_density_parameter
from Scripts.run_ss_on_exp_sim.scripts.summary_statistics.growth_rate_exp_deviation import get_exp_deviation, get_norm_growth_rate
from Scripts.run_ss_on_exp_sim.scripts.summary_statistics.convexity import cal_convexity
from Scripts.run_ss_on_exp_sim.scripts.helper_functions import helperFunctions 

import kl_divergence

cellmodeller_module = "simulation_module.py" # path to cellmodeller module
np.seterr(all="ignore") #suppress warnings for cleaner output
warnings.filterwarnings("ignore") # ignore all warnings

def model(parameters):
    """
    The "model" run by pyabc. Every time a simulation is run, the following occurs:
    1. Create a unique export path
    2. Run simulation and store data in export path
    3. Calculate and return summary statistics from the simulation
    
    The function must only have "parameters" as an argument to conform to pyabc's framework. 
    
    @param  parameters      dict of parameters specified in the prior distributions
    @return summary_stats   dict of calculated summary statistics
    """
    global max_cells
    global max_time
    global replicates
    
    # Extract time parameters from simulation (could hard-code to avoid unnecessary repetition)
    sim_file = os.path.abspath(cellmodeller_module)
    dt = sim_time_parameters(sim_file)
    
    # Storage variables
    summary_stats = {}
    aspect_ratio_list = []
    anisotropy_list = []
    density_list = []
    growth_rate_exp_deviation_list = []
    growth_rate_list = []
    convexity_list = []
    
    for n in range(replicates):
        # Defining input/output locations
        export_path = "data/" + str(uuid.uuid4())
        sys.stdout = open(os.devnull, 'w') # Disable printing from simulations
        try:    
            # Run CellModeller simulation
            sim = simulate(cellmodeller_module, parameters, export_path, max_cells=max_cells, max_time=max_time)
            sys.stdout = sys.__stdout__ # Re-enable printing
            print(f'{parameters} at {datetime.datetime.now()} complete; replicate {n}' )
        except:
            print(f'Simulation failed for parameters: {parameters}')
            #shututil.rmtree(export_path) 
                
        # Calculate summary stats                            
        try:
            # Load cellStates
            pickle_list = helperFunctions.create_pickle_list_full_path(export_path)
            cells = helperFunctions.load_cellStates_full_path(pickle_list[-1]) # load last pickle file
            # Calculate summary stats
            aspect_ratio = calc_aspect_ratio(cells)
            anisotropy = get_global_order_parameter(cells)
            density = get_density_parameter(cells)
            growth_rate_exp_deviation = get_exp_deviation(export_path, dt)
            growth_rate = get_norm_growth_rate(export_path, dt)
            convexity = cal_convexity(cells)
        except:
            # Stop repeating and assign unrealistic values to summary stats to reject
            print(f'Summary stat calculation failed: {export_path}')
            aspect_ratio_list = np.random.uniform(low=10, high=100, size=replicates)
            anisotropy_list = np.random.uniform(low=10, high=100, size=replicates)
            density_list = np.random.uniform(low=10, high=100, size=replicates)
            growth_rate_exp_deviation_list = np.random.uniform(low=10, high=100, size=replicates)
            growth_rate_list = np.random.uniform(low=10, high=100, size=replicates)
            convexity_list = np.random.uniform(low=10, high=100, size=replicates)
            cells = 0
            pickle_list = 0
            break 
      
        '''
        # Write summary stats to file for convenience (optional) 
        summary_stats_export = {'aspect_ratio': aspect_ratio,
                                'anisotropy': anisotropy,
                                'density': density,
                                'growth_rate_exp_deviation': growth_rate_exp_deviation,
                                'convexity': convexity,
                                'growth_rate': growth_rate}
        file = open(export_path + "/summary_stats.txt","w")
        for key, value in summary_stats_export.items(): 
            file.write('%s: %.3f\n' % (key, value)) 
        file.close()
        '''
         
        # Append summary stats to lists to create ensemble
        aspect_ratio_list.append(aspect_ratio)
        anisotropy_list.append(anisotropy)
        density_list.append(density)
        growth_rate_exp_deviation_list.append(growth_rate_exp_deviation)
        convexity_list.append(convexity)
        growth_rate_list.append(growth_rate)
    
    # Return summary statistics
    summary_stats["aspect_ratio"] = aspect_ratio_list
    summary_stats["order_parameter"] = anisotropy_list
    summary_stats["density_parameter"] = density_list
    summary_stats["R2_exp_fit"] = growth_rate_exp_deviation_list
    summary_stats['convexity'] = convexity_list
    summary_stats['growth_rate'] = growth_rate_list

    del aspect_ratio_list, anisotropy_list, density_list, growth_rate_exp_deviation_list, convexity_list, growth_rate_list, export_path, cells, pickle_list

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
            sim.step() # Could use try/except to delete sims if necessary
        
    # Export pickle at final timestep
    sim.writePickle()
    
    # Delete sim object to prevent OOMing?? I guess it works, but I got cell index error!?
    del sim
    
    gc.collect()
    
    return 0
    
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
       
def distance_calculation(sim_stats, exp_stats):
    """
    Calculate Euclidean distance between simulations and experimental observations
    
    @param  sim_stats   Dict of simulated summary statistics
    @param  exp_stats   Dict of experimental summary statistics
    @return distance    Euclidean distance between simuluations and experiment
    """
    print(sim_stats)
    try:
        distance_list = []
        for x in sim_stats.keys():
            divergence = kl_divergence.get_kl_divergence(exp_stats[x], sim_stats[x])
            distance_list.append(divergence)
        distance = np.linalg.norm(distance_list)
        print("Distance calc succeeded")
    except:
        print("Distance calculation failed")
        distance = 100 # Reject if there is an error
    
    gc.collect()

    return distance    
    
if __name__ == '__main__':
    # Define simulation termination condition; keep undesired option as None
    max_cells = 200
    max_time = None
    
    # Define number of simulation replicates
    replicates = 10
    
    # Define ABC-SMC settings
    n_cores = 6
    population_size = 40 # Number of simulations we need to accept to complete one calibration round
    min_epsilon = 0.05  # Stop if epsilon becomes smaller than this  
    max_populations = 5 #  Number of calibration rounds(stages)
    
    # Read in experimental data
    summary_stats = ['aspect_ratio', 'convexity', 'order_parameter', 'density_parameter', 'R2_exp_fit', 'growth_rate']
    exp_csv_file = 'exp_summary_stats.csv'
    exp_data = pd.read_csv(exp_csv_file)
    exp_summary_stats = {}
    for stat in summary_stats:
        exp_summary_stats[stat] = np.asarray(exp_data[stat])

    # Define prior distribution for each parameter [lb, ub]
    param_config = {'gamma': [np.log(0.1), np.log(1000)], 'reg_param': [np.log(0.01), np.log(100)]}
    
    prior_distributions = {}
    for parameter_name in param_config.keys():
        param_low = param_config[parameter_name][0]
        param_hi = param_config[parameter_name][1]
        width = abs(param_hi - param_low)

        # generate dictionary containing keywords for pyabc Distribution() class
        prior_distributions[parameter_name] = {"type": "uniform", "args": (param_low, width), "kwargs": {}}

    # create instance of `Distribution` class
    prior = pyabc.Distribution()
    prior = prior.from_dictionary_of_dictionaries(prior_distributions)
    
    # Create abc model
    # Initial_epsilon obtained from 40 particles, 10 replicates each with gamma = 1-1000, reg_param = 0.01-100
    #population_size=AdaptivePopulationSize(start_nr_particles=population_size, mean_cv=0.20), 
    abc = pyabc.ABCSMC(model, 
                       prior, 
                       distance_function=distance_calculation, 
                       population_size=population_size,
                       sampler=pyabc.sampler.MulticoreParticleParallelSampler(n_procs=n_cores),
                       #sampler=pyabc.sampler.MulticoreEvalParallelSampler(n_procs=n_cores),
                       #sampler=pyabc.sampler.SingleCoreSampler(),
                       #eps=pyabc.QuantileEpsilon(initial_epsilon=6.79301462, alpha=0.5, weighted=False)
                       eps=pyabc.QuantileEpsilon(alpha=0.5, weighted=False)
                       )
    
    # Create database                   
    db_path = "results.db"
    #abc.new("sqlite:///" + db_path, exp_summary_stats)
    abc.load("sqlite:///" + db_path, 1)
    
    # Run
    history = abc.run(minimum_epsilon=min_epsilon, 
                      max_nr_populations=max_populations) 
                      
