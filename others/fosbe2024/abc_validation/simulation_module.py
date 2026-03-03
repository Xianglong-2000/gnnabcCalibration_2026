import random
from CellModeller.Regulation.ModuleRegulator import ModuleRegulator
from CellModeller.Biophysics.BacterialModels.CLBacterium_reg_param import CLBacterium_reg_param
from CellModeller.GUI import Renderers
import numpy as np
from scipy.stats import lognorm

# Calibration parameters
gamma = 1000
reg_param = 0.1

# Physiological parameters
growth_mu = 1.15
growth_sigma = 0.31
division_mu = 4.80
division_sigma = 0.42
division_loc = -0.32
radius = 0.67
min_div_len = 2.0

dt = 0.025

def setup(sim):
    # Set biophysics module
    biophys = CLBacterium_reg_param(sim, jitter_z=False, max_cells=5000, reg_param=reg_param, gamma=gamma)

    # Set up regulation module
    regul = ModuleRegulator(sim, sim.moduleName)	
    # Only biophys and regulation
    sim.init(biophys, regul, None, None)
 
    # Specify the initial cell and its location in the simulation
    sim.addCell(cellType=0, pos=(0,0,0), dir=(1,0,0), rad=radius)

    # Add some objects to draw the models
    therenderer = Renderers.GLBacteriumRenderer(sim)
    sim.addRenderer(therenderer)
    
    # Specify how often data is saved
    sim.pickleSteps = 20
    sim.dt = dt #h

def init(cell):
    # Specify mean and distribution of initial cell size
    cell.targetVol = max(min_div_len, lognorm.rvs(division_sigma, loc=division_loc, scale=division_mu))
    cell.length = cell.targetVol/2
    cell.growthRate = np.random.normal(growth_mu, growth_sigma)
    cell.color = (0,0,0)

def update(cells):   
    #Iterate through each cell and flag cells that reach target size for division
    for (id, cell) in cells.items():
        max_strain_rate = np.exp(cell.growthRate*dt) - 1 # derived from exp growth and strainRate eq.
        if cell.cellAge >= 3:
            strain_rate_frac = 1 - cell.strainRate_rolling/max_strain_rate
            cell.color = (strain_rate_frac,strain_rate_frac,strain_rate_frac)

        if cell.volume > cell.targetVol:
            cell.divideFlag = True

def divide(parent, d1, d2):
    # Specify target cell size that triggers cell division
    d1.growthRate = np.random.normal(growth_mu, growth_sigma)
    d2.growthRate = np.random.normal(growth_mu, growth_sigma)
    
    d1.targetVol = max(min_div_len, lognorm.rvs(division_sigma, loc=division_loc, scale=division_mu))
    d2.targetVol = max(min_div_len, lognorm.rvs(division_sigma, loc=division_loc, scale=division_mu))

def setparams(param_dict):
    global gamma, reg_param
    gamma = np.exp(param_dict['gamma'])
    reg_param = np.exp(param_dict['reg_param'])

