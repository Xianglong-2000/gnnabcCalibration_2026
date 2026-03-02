import numpy as np
import pyabc
import torch
import statistics
import sys
import pickle as pk
import subprocess
import time
import warnings
warnings.filterwarnings("ignore")
import pickle

sys.path.append('D:/Projects/GNN Research/Code Files/my_code_files/s3')
import models_gatlearner

def model_test(parameters:dict):

    # subprocess the iteration to genereate and save graph data
    path = "D:/Projects/GNN Research/Code Files/my_code_files/s3/abcsmc_iter_gatlearner.py"
    envr = sys.executable
    time_limit = 3000
    try:
        subprocess.run([envr, path, str(parameters["gamma"]), str(parameters["reg_param"])],
                        check = True,
                        timeout = time_limit,  # Set timeout in seconds
                        )
    except subprocess.TimeoutExpired:
        print(f"Timeout after {time_limit}s — retrying...")
        subprocess.run([envr, path, str(parameters["gamma"]), str(parameters["reg_param"])],
                        check = True,
                        timeout = time_limit,  # Set timeout in seconds
                        )
    except subprocess.CalledProcessError as e:
        print(f"Script failed with error code {e.returncode}")
        subprocess.run([envr, path, str(parameters["gamma"]), str(parameters["reg_param"])],
                        check = True,
                        timeout = time_limit,  # Set timeout in seconds
                        )

    # load the saved graph data
    gamma = round(10**parameters["gamma"], 5)
    reg_param = round(10**parameters["reg_param"], 5)
    g_data_path = f"D:/Projects/GNN Research/Data Files/_simulator_graph_data_abc/{date_simulations}/abc_g_data_gamma={gamma}_reg_param={reg_param}.pkl"
    print(g_data_path)
    with open(g_data_path, "rb") as file:
        data = pk.load(file)

    # gnn prediction as the ss
    result = gnn_prediction(data)
    print("simulator ss: ", result)

    return result

def gnn_prediction(input_data) -> dict:
    """
    input = Data(x=[1681, 22], contact_edge_index=[2, 8346], lineage_edge_index=[2, 1440])
    output = {"gamma":100, "alpha":100}
    """

    # load the trained model
    model = models_gatlearner.GAT_Learner(in_channels=in_channels_model, 
                                          hidden_channels=hidden_channels_model,
                                          out_channels=feature_embedding_size)
    model.load_state_dict(torch.load(saved_model_path))
    model.eval()

    # predict graph embeddings
    with torch.no_grad():
        output = model(input_data)
    ##print("output: ",output)  ### check output structure
    #output_names = ["e1", "e2", "e3", "e4", "e5", "e6"]
    #result = {n:round(float(e),5) for n,e in zip(output_names,output.tolist()[0])}  ### in log10 space
    ##print("result: ",result)
    p1 = float(output.tolist()[0][0])
    p2 = float(output.tolist()[0][1])
    p3 = float(output.tolist()[0][2])
    p4 = float(output.tolist()[0][3])
    p5 = float(output.tolist()[0][4])
    p6 = float(output.tolist()[0][5])
    p1 = (p1 - min_ss1)/(max_ss1 - min_ss1)  ### min-max normalization
    p2 = (p2 - min_ss2)/(max_ss2 - min_ss2)  ### min-max normalization
    p3 = (p3 - min_ss3)/(max_ss3 - min_ss3)  ### min-max normalization
    p4 = (p4 - min_ss4)/(max_ss4 - min_ss4)  ### min-max normalization
    p5 = (p5 - min_ss5)/(max_ss5 - min_ss5)  ### min-max normalization
    p6 = (p6 - min_ss6)/(max_ss6 - min_ss6)  ### min-max normalization
    result = {"e1": p1,  ### this result is in log10 space
              "e2": p2,
              "e3": p3,
              "e4": p4,
              "e5": p5,
              "e6": p6,
              }  

    return result
   
def distance_calculation(sim_stats, exp_stats):
    distance_list = []
    for x in sim_stats.keys():
        diff = np.abs(sim_stats[x] - exp_stats[x])/exp_stats[x]
        distance_list.append(diff)
    distance = np.linalg.norm(distance_list)
    print("distance_list: ", distance_list)
    print("distance: ", distance)
    return distance 

if __name__ == '__main__':

    start_time_all = time.time()

    # set up local arguments to main() function
    date_simulations = "2025-09-25"
    moduleName = "simulation_module.py"
    in_channels_model = 31
    hidden_channels_model = 2*in_channels_model
    feature_embedding_size = 6
    CellTypes = ['YFP']
    dt_default = 0.025
    max_cells = 140
    max_time = None
    cell_type_mapping = {'YFP': 0}
    assign_cell_type = True
    use_grandmother_as_parent = False
    find_neighbors = True

    # get stacked mean and std from 1000 simulations for GNN
    parent_dir = f"D:/Projects/GNN Research/Data Files/_sim_mean_std_values_gnn/{date_simulations}/"  
    final_path = parent_dir + "CVIS_S3.pkl" 
    print(final_path) 
    with open(final_path, "rb") as file: 
        values = pk.load(file)  
    ##print(values)  
    gamma_mean = values["gamma mean"] 
    gamma_std = values["gamma std"]  
    reg_param_mean = values["reg_param mean"] 
    reg_param_std = values["reg_param std"] 
    mean_dict = values["feature mean"] 
    std_dict = values["feature std"]

    print("log(gamma) mean: ", gamma_mean)
    print("log(gamma) std: ", gamma_std)
    print("log(reg_param) mean: ", reg_param_mean)
    print("log(reg_param) std: ", reg_param_std)
    ##print("feature mean: ", mean_dict)
    ##print("feature std: ", std_dict)

    # get ss min and max from 1000 simulations for GNN
    parent_dir = f"D:/Projects/GNN Research/Data Files/_sim_mean_std_values_gnn/{date_simulations}/"  
    final_path = parent_dir + "CVIS_S3_minmaxss.pkl" 
    print(final_path) 
    with open(final_path, "rb") as file: 
        values = pk.load(file)  
    ##print(values)  
    min_ss1 = values["min ss1"] 
    min_ss2 = values["min ss2"] 
    min_ss3 = values["min ss3"] 
    min_ss4 = values["min ss4"] 
    min_ss5 = values["min ss5"] 
    min_ss6 = values["min ss6"] 
    max_ss1 = values["max ss1"] 
    max_ss2 = values["max ss2"] 
    max_ss3 = values["max ss3"] 
    max_ss4 = values["max ss4"] 
    max_ss5 = values["max ss5"] 
    max_ss6 = values["max ss6"] 
    print("min ss1: ", min_ss1)
    print("min ss2: ", min_ss2)
    print("min ss3: ", min_ss3)
    print("min ss4: ", min_ss4)
    print("min ss5: ", min_ss5)
    print("min ss6: ", min_ss6)
    print("max ss1: ", max_ss1)
    print("max ss2: ", max_ss2)
    print("max ss3: ", max_ss3)
    print("max ss4: ", max_ss4)
    print("max ss5: ", max_ss5)
    print("max ss6: ", max_ss6)

    saved_model_path = "D:/Projects/GNN Research/Data Files/_model_data_old/CVIS_S3_minmaxtest.pth"

    # 31 features
    include_columns = ['ImageNumber', #
                       'ObjectNumber', #
                       'id', #
                       'parent_id', #
                       'AreaShape_Area', #
                       'AreaShape_MajorAxisLength', #
                       'AreaShape_Orientation', #
                       'cellAge', #
                       'LifeHistory', #
                       'TrajectoryX', #
                       'TrajectoryY', #
                       'Direction_of_Motion', #
                       'Motion_Alignment_Angle', #
                       'Source_Neighbor_Avg_TrajectoryX', #
                       'Source_Neighbor_Avg_TrajectoryY', #
                       'Division_TimeStep', #
                       'Daughter_Mother_Length_Ratio', #
                       'Total_Daughter_Mother_Length_Ratio', #
                       'Max_Daughter_Mother_Length_Ratio', #
                       'Daughter_Avg_TrajectoryX', #
                       'Daughter_Avg_TrajectoryY', #
                       'Neighbor_Difference_Count', #
                       'Neighbor_Shared_Count', #
                       'Average_Length', #
                       'Velocity', #
                       'Instant_Velocity', #
                       'Average_Instant_Velocity', #
                       'Elongation_Rate', #
                       'Instant_Elongation_Rate', #
                       'startVol', #
                       'targetVol', #
                       'Prev_MajorAxisLength', #
                       'Bacterium_Movement', #
                       'dir_1', 'dir_2', #
                       ]
 
    ##parameters = {"gamma":math.log10(250), "reg_param":math.log10(30)}  ### for a simple test only
    ##model(parameters)  ### for a simple test only
    ##print()  ### for a simple test only

    # Step 0: load sim-exp g_data
    g_data_path = "D:/Projects/GNN Research/Data Files/_sim_graph_data_gnn/2025-08-24/CVIS_S3_simexp.pkl"
    with open(g_data_path, "rb") as file:
        data_tuple = pk.load(file)

    # Step 1: get sim-exp ss
    data_dict = {}
    for k, v in data_tuple.items():
        result = gnn_prediction(v)
        t = (v, result) 
        data_dict[k] = t

    e1_list = [v[-1]["e1"] for k, v in data_dict.items()]
    e2_list = [v[-1]["e2"] for k, v in data_dict.items()]
    e3_list = [v[-1]["e3"] for k, v in data_dict.items()]
    e4_list = [v[-1]["e4"] for k, v in data_dict.items()]
    e5_list = [v[-1]["e5"] for k, v in data_dict.items()]
    e6_list = [v[-1]["e6"] for k, v in data_dict.items()]

    exp_summary_stats = {}
    exp_summary_stats["e1"] = statistics.mean(e1_list)
    exp_summary_stats["e2"] = statistics.mean(e2_list)
    exp_summary_stats["e3"] = statistics.mean(e3_list)
    exp_summary_stats["e4"] = statistics.mean(e4_list)
    exp_summary_stats["e5"] = statistics.mean(e5_list)
    exp_summary_stats["e6"] = statistics.mean(e6_list)

    print("exp ss: ",exp_summary_stats)

    # Step 2: get prior
    param_config = {'gamma': [0, 3], 'reg_param': [0, 2]}  ### range of prior of log10 gamma and alpha
    prior_distributions = {}
    for parameter_name in param_config.keys():
        param_low = param_config[parameter_name][0]
        param_hi = param_config[parameter_name][1]
        width = abs(param_hi - param_low)
        prior_distributions[parameter_name] = {"type": "uniform", "args": (param_low, width), "kwargs": {}}
    prior = pyabc.Distribution.from_dictionary_of_dictionaries(prior_distributions)

    # Define ABC-SMC settings
    population_size = 100  
    min_epsilon = 0.12247
    max_populations = 100  

    # Step x: create database and abc model
    abc = pyabc.ABCSMC(model_test, 
                       prior, 
                       distance_calculation, 
                       population_size=population_size, 
                       sampler=pyabc.sampler.SingleCoreSampler(),  ### MulticoreEvalParallelSampler(n_cores) or SingleCoreSampler()
                       )
    """
    db_path = "CVIS_S3_minmaxtest.db"
    history = abc.new("sqlite:///D:/Projects/GNN Research/Data Files/abc_results/" + db_path, exp_summary_stats)
    print("ABC-SMC run ID:", history.id)
    """

    # to resume a stored run only
    #"""
    db_path = "CVIS_S3_minmaxtest.db"
    history = abc.load("sqlite:///D:/Projects/GNN Research/Data Files/abc_results/" + db_path, 1)  ## pick up where we were stopped, need to check id using jupyter
    print("num of completed populations: ", history.n_populations)
    print("ABC-SMC run ID: ", history.id)
    #"""

    # Step x: run
    history = abc.run(minimum_epsilon=min_epsilon, 
                      max_nr_populations=max_populations) 

    end_time_all = time.time()  ### about 20 hours for population_size*num_populations = 500
    print(f"Training time: {(end_time_all - start_time_all)/3600:.2f} hours")
