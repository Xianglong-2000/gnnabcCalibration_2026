import sys
sys.path.append('/work/x5bai/project/Code_Files/general/new_outputprocessing')
import os
import cellModellerProcessing
import time
import psutil
from multiprocessing import Pool

def runner(i):

    if i < 10:  # the date here is always the date when the pickle files were generated
        pkl_dir = f"/work/x5bai/project/Data_Files/_sim_pkl_data_gnn/{dt_sim}/iteration 00{i}/"  
    elif (i >= 10) and (i <100):
        pkl_dir = f"/work/x5bai/project/Data_Files/_sim_pkl_data_gnn/{dt_sim}/iteration 0{i}/"  
    elif (i >= 100) and (i <1000):
        pkl_dir = f"/work/x5bai/project/Data_Files/_sim_pkl_data_gnn/{dt_sim}/iteration {i}/"  
    else:
        print("We don't want the number of iterations greater than 999 since it's too computationally heavy")

    input_folder = pkl_dir + [f for f in os.listdir(pkl_dir)][0] + "/data"
    input_path = input_folder + "/" + [f for f in os.listdir(input_folder)][0]
    output_path = input_path.replace("_sim_pkl_data_gnn","_sim_csv_data_gnn")
    os.makedirs(output_path, exist_ok=True)
    print("in: ", input_path)
    print("out: ", output_path)

    try:
        cellModellerProcessing.process_simulation_directory(input_directory=input_path, cell_type_mapping=cell_type_mapping,
                                                                    output_directory=output_path, assign_cell_type=assign_cell_type,
                                                                    use_grandmother_as_parent=use_grandmother_as_parent,
                                                                    find_neighbors=find_neighbors, pixel_per_micron=pixel_per_micron,
                                                                    cellprofiler_orientation_format=cellprofiler_orientation_format)
    except Exception as e:
        error_dir = f"/work/x5bai/project/Data_Files/_errors/OP/{dt_sim}/{[f for f in os.listdir(pkl_dir)][0]}"
        os.makedirs(error_dir, exist_ok=True)

        return runner(i)    

if __name__ == '__main__':

    # start time
    start_time = time.time()

    # If True, infer and assign cell types to tracked bacteria.
    assign_cell_type = True

    # Dictionary mapping cell type names to CellModeller IDs.
    cell_type_mapping = {'YFP': 0}

    # If True, approximates the parent bacterium by selecting the nearest disappeared bacterium from
    # the previous time step. Useful when large time step gaps cause the actual parent to no longer appear in
    # the previous step.
    use_grandmother_as_parent = False

    # Computes neighbor relationships between bacteria based on spatial proximity.
    # Two bacteria are considered neighbors if their expanded pixel boundaries touch.
    # This is consistent with CellProfiler's "MeasureObjectNeighbors" module.
    find_neighbors = True

    # If the unit of length and the center coordinates of the bacteria are in µm, you need to pass this variable
    # to convert them into pixels, ensuring that the output is suitable for use as input in TrackRefiner.
    pixel_per_micron = 0.144

    # If True, converts CellModeller orientation angles into CellProfiler’s AreaShape_Orientation convention.
    cellprofiler_orientation_format = True

    num_iter = 1000
    dt_sim = "2025-11-22"

    indices = list(range(num_iter))

    with Pool(processes=10) as pool:
        results = pool.map(runner, indices)

    # end time
    end_time = time.time()
    duration = (end_time - start_time)/(60*60)
    print(f"Computing time: {duration:.3f} hours")