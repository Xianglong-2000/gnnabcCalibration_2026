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
import os
import re
import sys
import uuid

import warnings
warnings.filterwarnings("ignore")

# Installed modules
from CellModeller.Simulator import Simulator
##import pyabc

sys.path.append('D:/Projects/Github Packages/CellModeller-ingallslab')
from Scripts.run_ss_on_exp_sim.scripts.summary_statistics.AspectRatio import calc_aspect_ratio
from Scripts.run_ss_on_exp_sim.scripts.summary_statistics.Anisotropy import get_global_order_parameter
from Scripts.run_ss_on_exp_sim.scripts.summary_statistics.density_calculation import get_density_parameter
from Scripts.run_ss_on_exp_sim.scripts.summary_statistics.growth_rate_exp_deviation import get_exp_deviation, get_norm_growth_rate
from Scripts.run_ss_on_exp_sim.scripts.summary_statistics.convexity_smart import cal_convexity

from Scripts.run_ss_on_exp_sim.scripts.helper_functions import helperFunctions 

cellmodeller_module = "simulation_module.py"  


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
    # Defining input/output locations
    export_path = "data/" + str(uuid.uuid4())
    ##sys.stdout = open(os.devnull, 'w') # Disable printing from simulations
    
    # Run CellModeller simulation
    simulate(cellmodeller_module, parameters, export_path, max_cells=max_cells, max_time=max_time)
    ##sys.stdout = sys.__stdout__ # Re-enable printing
    print("Simulation completed")
    
    # Load cellStates
    pickle_list = helperFunctions.create_pickle_list_full_path(export_path)
    cells = helperFunctions.load_cellStates_full_path(pickle_list[-1]) # load last pickle file
    
    # Extract time parameters from simulation (could hard-code to avoid unnecessary repetition)
    sim_file = os.path.abspath(cellmodeller_module)
    dt = sim_time_parameters(sim_file)
    
    # Calculate summary statistics
    summary_stats = {}
    summary_stats["Aspect ratio"] = calc_aspect_ratio(cells)
    summary_stats['Convexity'] = cal_convexity(cells)
    summary_stats["Order parameter"] = get_global_order_parameter(cells)
    summary_stats["Density"] = get_density_parameter(cells)
    summary_stats["Agreement with exponential growth"] = get_exp_deviation(export_path, dt)
    summary_stats["Normalized growth rate"] = get_norm_growth_rate(export_path, dt, n_max=225, t_min=72*3/60)


    # Write summary stats to file for convenience (optional) 
    ##file = open(export_path + "/summary_stats.txt","w")
    ##for key, value in summary_stats.items(): 
    ##    file.write('%s: %.3f\n' % (key, value)) 
    ##file.close()

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
       
def distance_calculation(sim_stats, exp_stats):
    """
    Calculate Euclidean distance between simulations and experimental observations
    
    @param  sim_stats   Dict of simulated summary statistics
    @param  exp_stats   Dict of experimental summary statistics
    @return distance    Euclidean distance between simuluations and experiment
    """
    distance_list = []
    for x in sim_stats.keys():
        diff = np.abs(sim_stats[x] - exp_stats[x])/exp_stats[x]
        distance_list.append(diff)

    distance = np.linalg.norm(distance_list)

    return distance    
    
if __name__ == '__main__':
    # Define simulation termination condition; keep undesired option as None
    max_cells = 100
    max_time = None

    # simple test
    gamma = 100
    reg_param = 5

    parameters = {"gamma":gamma, "reg_param":reg_param}

    print(parameters)
    model(parameters)

    breakpoint()


    """
    # Define ABC-SMC settings
    n_cores = 8
    population_size = 10 #40 # Number of simulations we need to accept to complete one calibration round
    min_epsilon = 0.05  # Stop if epsilon becomes smaller than this  
    max_populations = 1 #5 #  Number of calibration rounds(stages)

    # Define experimental data
    exp_summary_stats = {}
    exp_summary_stats["Aspect Ratio"] = 0.706879319359634
    exp_summary_stats["Anisotropy"] = 0.75535426433106
    exp_summary_stats["Density"] = 0.953001564059211
    exp_summary_stats["growth_rate_exp_deviation"] = 0.985259710208264
    exp_summary_stats["fourier_descriptor"] = 0.0206358837797498

    # Define prior distribution for each parameter [lb, ub]
    param_config = {'gamma': [10, 1000], 'reg_param': [1/1000, 1/10]}
    
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
    abc = pyabc.ABCSMC(model, 
                       prior, 
                       distance_function=distance_calculation, 
                       population_size=population_size, 
                       #sampler=pyabc.sampler.MulticoreEvalParallelSampler(n_cores)
                       )
    
    # Create database                   
    db_path = "results.db"
    abc.new("sqlite:///" + db_path, exp_summary_stats)
    #abc.load("sqlite:///" + db_path, 1)
    
    # Run
    history = abc.run(minimum_epsilon=min_epsilon, 
                      max_nr_populations=max_populations) 
    """