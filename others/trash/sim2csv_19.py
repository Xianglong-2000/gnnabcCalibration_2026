import sys
sys.path.append('C:/Users/MECHREV/CellModeller-ingallslab')
import os
import output_processing.CellModellerProcessing

# The name of the bacteria Types.
CellTypes = ['YFP']

num_iter = 1
for i in range(num_iter):
    if i < 10:
        parent_dir = f"D:/Research/Data Files/_sim_exp_pkl_data_abc/iteration 0{i}/"
        new_dir = f"D:/Research/Data Files/_sim_exp_csv_data_abc/iteration 0{i}"
    elif (i >= 10) and (i <100):
        parent_dir = f"D:/Research/Data Files/_sim_exp_pkl_data_abc/iteration {i}/"
        new_dir = f"D:/Research/Data Files/_sim_exp_csv_data_abc/iteration {i}"
    else:
        print("We don't want the number of iterations greater than 99 since it's too computationally heavy")

    os.makedirs(new_dir, exist_ok=True)  # create a new folder if it doesn't exist and do nothing if it exists

    folders = [f for f in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, f))]
    for f in folders:
        sub_parent_dir = parent_dir + f + "/data"
        folders = [f for f in os.listdir(sub_parent_dir) if os.path.isdir(os.path.join(sub_parent_dir, f))]
        folder = folders[0]

        # Getting the location of simulations.
        input_dir = sub_parent_dir + "/" + folder

        # The location of the CSV output files
        output_dir = new_dir + "/" + f
        os.makedirs(output_dir, exist_ok=True)  # create a new folder if it doesn't exist and do nothing if it exists

        # Start Processing
        output_processing.CellModellerProcessing.starting_process(input_dir, CellTypes, output_dir)

