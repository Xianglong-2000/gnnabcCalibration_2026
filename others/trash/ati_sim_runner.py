
def simulation_script(script_name, gama_val, reg_param_val, growth_rate):

    script = """
import random
from CellModeller.Regulation.ModuleRegulator import ModuleRegulator
from CellModeller.Biophysics.BacterialModels.CLBacterium_reg_param import CLBacterium_reg_param
from CellModeller.GUI import Renderers
import numpy
from scipy.stats import lognorm


def setup(sim):
    sim.dt = 0.025
    # Set biophysics module
    biophys = CLBacterium_reg_param(sim, jitter_z=False, gamma=""" + str(gama_val) + """, reg_param=""" + str(reg_param_val) + """)

    # Set up regulation module
    regul = ModuleRegulator(sim, sim.moduleName)
    # Only biophysics and regulation
    sim.init(biophys, regul, None, None)

    # Specify the initial cell and its location in the simulation
    sim.addCell(cellType=0, pos=(0,0,0), dir=(1,0,0))

    # Add some objects to draw the models
    therenderer = Renderers.GLBacteriumRenderer(sim)
    sim.addRenderer(therenderer)

    # Specify how often data is saved
    sim.pickleSteps = 10
    sim.saveOutput = True


def init(cell):
    # Specify mean and distribution of initial cell size
    cell.targetVol = 3.5 + random.uniform(0.0,0.5)
    # Specify growth rate of cells
    cell.growthRate = """ + str(growth_rate) + """
    cell.color = (0.0,1.0,0.0)

def update(cells):
    #Iterate through each cell and flag cells that reach target size for division
    for (id, cell) in cells.items():
        if cell.volume > cell.targetVol:
            cell.divideFlag = True

def divide(parent, d1, d2):
    # Specify target cell size that triggers cell division
    d1.targetVol = 3.5 + random.uniform(0.0,0.5)
    d2.targetVol = 3.5 + random.uniform(0.0,0.5)

    """

    # write script
    f = open("scripts/" + script_name + ".py", "w")
    f.write(script)
    f.close()
