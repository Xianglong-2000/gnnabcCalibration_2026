import subprocess
import time
import os
from pathlib import Path

import warnings
warnings.filterwarnings("ignore")


if __name__ == "__main__":

   # test_cli_with_args("D:/Projects/GNN Research/Data Files/000")
    # 接下来做一个loop类似之前做的就ok了

    start_time = time.time()

    num_iter = 31
    dt_sim = "2025-10-20"
    # iterations = [5,6,7,8,9]

    # for i in iterations:
    for i in range(num_iter):
        if i < 10:  # the date here is always the date when the pickle files were generated
            parent_dir = f"D:/Projects/GNN Research/Data Files/_sim_pre_tr_csv_data_gnn/{dt_sim}/iteration 00{i}/"  # pickle folder path
            new_dir = f"D:/Projects/GNN Research/Data Files/_sim_tr_csv_data_gnn/{dt_sim}/iteration 00{i}"  # csv folder path
        elif (i >= 10) and (i < 100):
            parent_dir = f"D:/Projects/GNN Research/Data Files/_sim_pre_tr_csv_data_gnn/{dt_sim}/iteration 0{i}/"  # pickle folder path
            new_dir = f"D:/Projects/GNN Research/Data Files/_sim_tr_csv_data_gnn/{dt_sim}/iteration 0{i}"  # csv folder path
        elif (i >= 100) and (i < 1000):
            parent_dir = f"D:/Projects/GNN Research/Data Files/_sim_pre_tr_csv_data_gnn/{dt_sim}/iteration {i}/"  # pickle folder path
            new_dir = f"D:/Projects/GNN Research/Data Files/_sim_tr_csv_data_gnn/{dt_sim}/iteration {i}"  # csv folder path
        else:
            print("We don't want the number of iterations greater than 999 since it's too computationally heavy")

        os.makedirs(new_dir, exist_ok=True)  # create a new folder if it doesn't exist and do nothing if it exists

        folders = [f for f in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, f))]
        # folders = ["gamma=43.85314_reg_param=5.2315_iter=0"]  # for debugging

        for f in folders:
            print(f)
            input_dir = parent_dir + f
            print(input_dir)

            # The location of the CSV output files
            output_dir = Path(new_dir + "/" + f)
            os.makedirs(output_dir, exist_ok=True)  # create a new folder if it doesn't exist and do nothing if it exists
            env = os.environ.copy()
            env["PYTHONUTF8"] = "1"

            # Start Processing
            # Simulate running the CLI with valid arguments
            result = subprocess.run(
                ["D:/Projects/GNN Research/Code Files/myenv/Scripts/python.exe",
                "-m", "Trackrefiner.cli",
                "-i", input_dir + "/Objects properties.csv",
                "-n", input_dir + "/Object relationships.csv",
                "-t", "3",
                "-d", "20",
                "-p", "0.144",
                "-o", output_dir,
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
            ##print("STDOUT:", result.stdout)
            ##print("STDERR:", result.stderr)

            assert result.returncode == 0  # Ensure the process exits successfully
            # Check for the final success log message in stdout
            assert "Trackrefiner Process completed at:" in result.stdout, (
                "Expected log message indicating successful completion was not found in stdout."
            )
            assert output_dir.exists()  # Check if the output directory was created

    # check the training duration
    end_time = time.time()
    duration = (end_time - start_time) / (60 * 60)
    print(f"Computing time: {duration:.3f} hours")