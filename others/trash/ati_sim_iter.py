import sys
from CellModeller.Simulator import Simulator
import os
import numpy as np
import random
import string
import pandas as pd
from CreateSimulationScript import simulation_script


def simulate(modfilename, platform, device, output_name):
    (path, name) = os.path.split(modfilename)
    modname = str(name).split('.')[0]
    sys.path.append(path)
    sim = Simulator(modname, Sim_dt, outputDirName=output_name, clPlatformNum=platform, clDeviceNum=device,
                    saveOutput=True)
    while len(sim.cellStates) <= sim_max_num_cells_in_last_time_step:
        sim.step()


def run_simulation(gama_val, reg_param_val, growth_rate):
    # Execute CellModeller simulation with given parameters
    script_name = ''.join(
        random.SystemRandom().choice(string.ascii_uppercase + string.digits) for _ in range(6))
    print(script_name)
    simulation_script(script_name, gama_val, reg_param_val, growth_rate)

    script_path = "scripts/" + script_name + ".py"
    simulate(script_path, 0, 0, script_name)
    print("Finished Simulation")


if __name__ == '__main__':

    # create directories
    simulation_results_path = "data"
    scripts_path = "scripts"
    if not os.path.exists(simulation_results_path):
        os.makedirs(simulation_results_path)
    if not os.path.exists(scripts_path):
        os.makedirs(scripts_path)

    global Sim_dt
    global sim_max_num_cells_in_last_time_step
    global summary_statistic_method_list
    Sim_dt = 0.025
    sim_max_num_cells_in_last_time_step = 1000

    print("Running Simulation")

    # parameter distribution
    param_distributions = {'gamma': [1, 10, 100, 1000]}

    for gama_val in [1, 10, 100, 1000]:
        for reg_param_val in [0.05, 0.1, 0.5, 1]:
            for growth_rate in [2.5, 3, 4]:
                run_simulation(gama_val, reg_param_val, growth_rate)
