import subprocess
import sys
import time

import warnings
warnings.filterwarnings("ignore")

def main() -> None:

    start_time = time.time()

    num_iterations = 1000 # number of iterations we want

    for i in range(num_iterations):
    ##for i in range(932, 933):    
        print(f"\n--- Running iteration {i} ---")
        path = "D:/Projects/GNN Research/Code Files/my_code_files/s3/sim_iter_learner.py"
        envr = sys.executable
        subprocess.run([envr, path, str(i)],
                        timeout = 1500,  # Set timeout in seconds
                        )

    print("-- All Simulations Completed --")

    # check the training duration
    end_time = time.time()
    duration = (end_time - start_time)/(60*60)
    print(f"Computing time: {duration:.3f} hours")

if __name__ == "__main__": 
    main()