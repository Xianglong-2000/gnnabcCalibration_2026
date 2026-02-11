import sys
sys.path.append('../')
import CellModellerProcessing

# Getting the location of simulations.
Input = "../input/cell modeller data"
# The name of the bacteria Types.
CellTypes = ['YFP']
# The location of the CSV output files
Output = "../output/"

# Start Processing
CellModellerProcessing.starting_process(Input, CellTypes, Output)
