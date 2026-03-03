import subprocess
import time
import os
from pathlib import Path
import psutil
from multiprocessing import Pool

import warnings
warnings.filterwarnings("ignore")

def runner(i):

    if i < 10:  # the date here is always the date when the pickle files were generated
        csv_dir = f"/work/x5bai/project/Data_Files/_sim_csv_data_gnn/{dt_sim}/iteration 00{i}/"  
    elif (i >= 10) and (i <100):
        csv_dir = f"/work/x5bai/project/Data_Files/_sim_csv_data_gnn/{dt_sim}/iteration 0{i}/"  
    elif (i >= 100) and (i <1000):
        csv_dir = f"/work/x5bai/project/Data_Files/_sim_csv_data_gnn/{dt_sim}/iteration {i}/"  
    else:
        print("We don't want the number of iterations greater than 999 since it's too computationally heavy")

    input_folder = csv_dir + [f for f in os.listdir(csv_dir)][0] + "/data"
    input_path = input_folder + "/" + [f for f in os.listdir(input_folder)][0]
    output_path = input_path.replace("_sim_csv_data_gnn","_sim_tr_csv_data_gnn")
    os.makedirs(output_path, exist_ok=True)
    print("in: ", input_path)
    print("out: ", output_path)

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"

    try:
        result = subprocess.run(
                ["/work/x5bai/miniconda3/envs/project2/bin/python",
                "-m", "Trackrefiner.cli",
                "-i", input_path + "/Objects_properties.csv",
                "-n", input_path + "/Object_relationships.csv",
                "-t", "3",
                "-d", "20",
                "-p", "0.144",
                "-o", output_path,
                ## '--elongation_rate_method'; default="Average"
                ## '--intensity_threshold'; default=0.1
                ## '--classifier'; default='LogisticRegression'
                ## '--num_cpus'; default=-1
                "--disable_tracking_correction", ### '--disable_tracking_correction'; Default: Disabled
                "--save_pickle",  ### '--save_pickle'; Default: Disabled
                ],
                text=True,
                capture_output=True,
                encoding="utf-8", 
                env=env)

        # debug
        """
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)

        assert result.returncode == 0  # Ensure the process exits successfully
        # Check for the final success log message in stdout
        assert "Trackrefiner Process completed at:" in result.stdout, (
            "Expected log message indicating successful completion was not found in stdout."
        )
        assert output_dir.exists()  # Check if the output directory was created
        """
        
    except Exception as e:
        error_dir = f"/work/x5bai/project/Data_Files/_errors/TR/{dt_sim}/{[f for f in os.listdir(csv_dir)][0]}"
        os.makedirs(error_dir, exist_ok=True)

        return runner(i)    

if __name__ == "__main__":

    # start time
    start_time = time.time()

    num_iter = 1000
    dt_sim = "2025-11-22"

    indices = list(range(num_iter))

    with Pool(processes=10) as pool:
        results = pool.map(runner, indices)

    # check the training duration
    end_time = time.time()
    duration = (end_time - start_time) / (60 * 60)
    print(f"Computing time: {duration:.3f} hours")