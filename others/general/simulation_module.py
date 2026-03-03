from CellModeller.Regulation.ModuleRegulator import ModuleRegulator
from CellModeller.Biophysics.BacterialModels.CLBacterium_reg_param import CLBacterium_reg_param
from CellModeller.GUI import Renderers
from scipy import stats
import numpy as np

# Calibration parameters
gamma = 100
reg_param = 0.01 

# Physiological parameters
growth_mu = 1.15
growth_sigma = 0.31

division_shape = 0.42
division_scale = 4.8
division_local = -0.32

radius = 0.67

def setup(sim):
    # Set biophysics module
    biophys = CLBacterium_reg_param(sim, jitter_z=False, max_cells=5000, reg_param=reg_param, gamma=gamma)
    print(f"gamma is {gamma}, reg_param is {reg_param}")

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
    sim.pickleSteps = 2
    sim.dt = 0.025 #h

def init(cell):
    # Specify mean and distribution of initial cell size
    cell.targetVol = stats.lognorm.rvs(s=division_shape, loc=division_local, scale=division_scale)

    # Specify growth rate of cells
    cell.growthRate = np.random.normal(growth_mu, growth_sigma)

    cell.color = (0.0,1.0,0.0)

def update(sim,cells):
    #Iterate through each cell and flag cells that reach target size for division
    for (id, cell) in cells.items():
        if cell.strainRate < 0.01:
            cell.color = (1,0,0)
        else:
            cell.color = (0,1,0)
        if cell.volume > cell.targetVol:
            cell.divideFlag = True

def divide(parent, d1, d2):
    # Specify target cell size that triggers cell division
    d1.targetVol = stats.lognorm.rvs(s=division_shape, loc=division_local, scale=division_scale)
    d2.targetVol = stats.lognorm.rvs(s=division_shape, loc=division_local, scale=division_scale)
    d1.growthRate = np.random.normal(growth_mu, growth_sigma)
    d2.growthRate = np.random.normal(growth_mu, growth_sigma)
    
def setparams(param_dict):
    global gamma, reg_param
    gamma = param_dict['gamma']
    reg_param = param_dict['reg_param']
