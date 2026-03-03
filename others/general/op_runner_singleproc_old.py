import sys
sys.path.append('D:/Projects/Github Packages/CellModeller-ingallslab/output-processing')
import os
import CellModellerProcessing
import time
import psutil

def main():

    # Dictionary mapping cell type names to CellModeller IDs.
    cell_type_mapping = {'YFP': 0}

    # If True, infer and assign cell types to tracked bacteria.
    assign_cell_type = True

    # If True, approximates the parent bacterium by selecting the nearest disappeared bacterium from
    # the previous time step. Useful when large time step gaps cause the actual parent to no longer appear in
    # the previous step.
    use_grandmother_as_parent = False

    # Computes neighbor relationships between bacteria based on spatial proximity.
    # Two bacteria are considered neighbors if their expanded pixel boundaries touch.
    # This is consistent with CellProfiler's "MeasureObjectNeighbors" module.
    find_neighbors = True

    num_iter = 31
    dt_sim = "2025-10-20"

    ram = psutil.virtual_memory()
    total_gb = ram.total / (1024 ** 3)
    print(f"Total RAM: {total_gb:.2f} GB")
    print()

    start_time = time.time()

    for i in range(num_iter):
    ##for i in range(676, num_iter):
        if i < 10:  # the date here is always the date when the pickle files were generated
            parent_dir = f"D:/Projects/GNN Research/Data Files/_sim_pkl_data_gnn/{dt_sim}/iteration 00{i}/"  # pickle folder path
            new_dir = f"D:/Projects/GNN Research/Data Files/_sim_csv_data_gnn/{dt_sim}/iteration 00{i}"  # csv folder path
        elif (i >= 10) and (i <100):
            parent_dir = f"D:/Projects/GNN Research/Data Files/_sim_pkl_data_gnn/{dt_sim}/iteration 0{i}/"  # pickle folder path
            new_dir = f"D:/Projects/GNN Research/Data Files/_sim_csv_data_gnn/{dt_sim}/iteration 0{i}"  # csv folder path
        elif (i >= 100) and (i <1000):
            parent_dir = f"D:/Projects/GNN Research/Data Files/_sim_pkl_data_gnn/{dt_sim}/iteration {i}/"  # pickle folder path
            new_dir = f"D:/Projects/GNN Research/Data Files/_sim_csv_data_gnn/{dt_sim}/iteration {i}"  # csv folder path
        else:
            print("We don't want the number of iterations greater than 999 since it's too computationally heavy")

        os.makedirs(new_dir, exist_ok=True)  # create a new folder if it doesn't exist and do nothing if it exists

        folders = [f for f in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, f))]
        # folder_with_error = [folders[25],folders[26],folders[27],folders[28],folders[29]]

        num_skipped = 0  # count the number of skipped simulations which would cause the memory issue

        for f in folders:
            print(f"Processing {f}")
            sub_parent_dir = parent_dir + f + "/data"
            folders = [f for f in os.listdir(sub_parent_dir) if os.path.isdir(os.path.join(sub_parent_dir, f))]
            folder = folders[0]

            # Getting the location of simulations.
            input_dir = sub_parent_dir + "/" + folder

            # The location of the CSV output files
            output_dir = new_dir + "/" + f
            os.makedirs(output_dir, exist_ok=True)  # create a new folder if it doesn't exist and do nothing if it exists

            # print(input_dir)

            # Start Processing
            try:
                CellModellerProcessing.process_simulation_directory(
                input_directory=input_dir,
                cell_type_mapping=cell_type_mapping,
                output_directory=output_dir,
                assign_cell_type=assign_cell_type,
                use_grandmother_as_parent=use_grandmother_as_parent,
                find_neighbors=find_neighbors
                )
            except MemoryError:  # skip the output process for simulations which cause memory issue
                print(f"MemoryError occurred at iteration {i}, simulation f, skipping...")
                if os.path.exists(output_dir + "/Objects properties.csv"):
                    os.remove(output_dir + "/Objects properties.csv")
                elif os.path.exists(output_dir + "/Object relationships.csv"):
                    os.remove(output_dir + "/Object relationships.csv")
                os.rmdir(output_dir)
                num_skipped += 1
                continue

            print("Done")
            print()

        print(f"There are {num_skipped} skipped simulations in iteration {i}")
        print()

    # check the training duration
    end_time = time.time()
    duration = (end_time - start_time)/(60*60)
    print(f"Computing time: {duration:.3f} hours")

if __name__ == '__main__':
    main()