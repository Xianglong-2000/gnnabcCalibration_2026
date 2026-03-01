import numpy as np
import pandas as pd
import pyabc
import torch
from torch_geometric.data import Data
import re
import math
from CellModeller.Simulator import Simulator
import statistics
import networkx as nx
import os
import sys
import pickle as pk
from pathlib import Path
import subprocess
import shutil
import time
import warnings
warnings.filterwarnings("ignore")

from torch.utils.data import DataLoader as TorchDataLoader
from torch_geometric.data import Batch

sys.path.append('D:/Projects/GNN Research/Code Files/my_code_files/s2')
import models_gatlstmregressor


def samples_to_sequences(file_path):  ### input = D:/Projects/GNN Research/Data Files/_sim_graph_splitted_data_gnn/2025-08-26
    folders = [os.path.join(file_path, f) for f in os.listdir(file_path) if os.path.isdir(os.path.join(file_path, f))]  ### .../iteration 000/
    data_list = []
    for sample_path in folders:
        l1_folder = [os.path.join(sample_path, f) for f in os.listdir(sample_path) if
                     os.path.isdir(os.path.join(sample_path, f))]  ### .../gamma=37.6_reg_param=1.24_iter=0/
        l2_folder = [os.path.join(l1_folder[0], f) for f in os.listdir(l1_folder[0]) if
                     os.path.isdir(os.path.join(l1_folder[0], f))]  ### .../node_feature_data/
        pkl_file_path = [os.path.join(l2_folder[0], f) for f in os.listdir(l2_folder[0])][0]  ### .../test0826_1000sims_stdnorm_log10_27f.pkl
        with open(pkl_file_path, "rb") as file:
            seq_dict = pk.load(file)  ### now the pickle is loaded

        y_tensor = None
        data = []
        indices = []
        for k, v in seq_dict.items():  ### now need to match it to the same structure as train_ds in test2()
            if hasattr(v, "y"):
                if y_tensor is None:
                    y_tensor = v.y
                del v.y
            data.append(v)
            idx = int(k.split("=")[-1].split(".")[0])
            indices.append(idx)
        pairs = sorted(zip(indices, data), key=lambda x: x[0])  ### making sure indices are increasing is enough and they will take data only
        ##print(pairs)  ### number should be increasing although the first number varies
        data = [comp for _, comp in pairs]  ### make sure sequence is in the right order
        ##print(idx for idx, _ in pairs)  ### it's supposed to be increasing
        data_tuple = (data, y_tensor)  ### ([Data1, ..., Data5], y) for each sample
        data_list.append(data_tuple)  ### [Sample1, Sample2, ..., Sample10] across all 1000 samples

    return data_list


def collate_sequences_batch(batch):
    if type(batch) == tuple:
        batch = [batch]

    sequences = []
    y_seqs = []

    # Unpack outer (seq, y_seq) or just seq
    for item in batch:
        seq, y_seq = item
        sequences.append(list(seq))
        y_seqs.append(y_seq if isinstance(y_seq, torch.Tensor) else torch.as_tensor(y_seq, dtype=torch.float))

    flat_graphs: List[Data] = []
    seq_ids_: List[int] = []
    t_steps_: List[int] = []

    for seq_idx, seq in enumerate(sequences):
        for t_idx, elem in enumerate(seq):
            g, t = elem, t_idx
            flat_graphs.append(g)
            seq_ids_.append(seq_idx)
            t_steps_.append(int(t))

    flat_batch = Batch.from_data_list(flat_graphs)
    seq_ids = torch.tensor(seq_ids_, dtype=torch.long)
    t_steps = torch.tensor(t_steps_, dtype=torch.long)

    y_seq = None
    if y_seqs:
        y_seq = torch.stack(y_seqs).to(torch.float)  # [B, 2]

    return flat_batch, seq_ids, t_steps, y_seq


def model_test(parameters:dict):

    # subprocess the iteration to genereate and save graph data
    path = "D:/Projects/GNN Research/Code Files/my_code_files/s2/abcsmc_iter_gatlstmregressor.py"
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
    graph_id = f"gamma={gamma}_reg_param={reg_param}"

    pkl_file_path = f"D:/Projects/GNN Research/Data Files/_simulator_graph_splitted_data_abc/2025-08-20/{graph_id}"
    with open(pkl_file_path + "/CVIS_S2.pkl", "rb") as file:
        seq_dict = pk.load(file)  ### now the pickle is loaded
    y_tensor = None
    data = []
    indices = []
    for k, v in seq_dict.items():  ### now need to match it to the same structure as train_ds in test2()
        if hasattr(v, "y"):
            if y_tensor is None:
                y_tensor = v.y
            del v.y
        data.append(v)
        idx = int(k.split("=")[-1].split(".")[0])
        indices.append(idx)
    pairs = sorted(zip(indices, data),
                   key=lambda x: x[0])  ### making sure indices are increasing is enough and they will take data only
    ##print(pairs)  ### number should be increasing although the first number varies
    data = [comp for _, comp in pairs]  ### make sure sequence is in the right order
    ##print(idx for idx, _ in pairs)  ### it's supposed to be increasing
    data_tuple = (data, y_tensor)  ### ([Data1, ..., Data5], y) for each sample

    data_list = [data_tuple]
    loader = TorchDataLoader(data_list, batch_size=1, shuffle=False, collate_fn=collate_sequences_batch)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models_gatlstmregressor.GAT_LSTM_Regressor(in_dim=num_features,
                                        gnn_hidden_dim=2*num_features,
                                        z_dim=2*num_features,
                                        lstm_hidden_dim=2*num_features,
                                        out_dim=num_params).to(device)
    model.load_state_dict(torch.load(saved_model_path))
    model.eval()

    # predict gamma and reg_param
    with torch.no_grad():
        for flat_batch, seq_ids, t_steps, y_seq in loader:
            flat_batch = flat_batch.to(device)
            seq_ids = seq_ids.to(device)
            t_steps = t_steps.to(device)
            y_seq = y_seq.to(device)  # [B, 2]

            preds, lengths = model(flat_batch, seq_ids, t_steps)  # preds: [B, 2]
            print("targets: ", y_seq)
            print("predictions: ", preds)
            print("num of seq sub-graphs in each sample: ", lengths)
            print()

    gamma_list = [round(float(param[0] * gamma_std + gamma_mean), 5) for param in preds.tolist()]  ### preds = tensor([[g,a]])
    alpha_list = [round(float(param[1] * reg_param_std + reg_param_mean), 5) for param in preds.tolist()]
    p1 = gamma_list[0]  ### de-normalized
    p2 = alpha_list[0]  ### de-normalized
    p1 = (p1-gamma_min)/(gamma_max-gamma_min)  ### min-max normalization
    p2 = (p2-reg_param_min)/(reg_param_max-reg_param_min)  ### min-max normalization
    result = {"gamma": p1, "alpha": p2}  ### this result is in log10 space
    print("parameter prediction: ", {k:round(10**v, 5) for k,v in result.items()})
    print("log10 parameter prediction: ", result)
    return result

   
def distance_calculation(sim_stats, exp_stats):
    distance_list = []
    for x in sim_stats.keys():
        std_exp_stats = np.where(exp_stats[x] == 0, 1e-8, exp_stats[x])
        diff = np.abs(sim_stats[x] - exp_stats[x])/std_exp_stats
        distance_list.append(diff)
    distance = np.linalg.norm(distance_list)
    return distance 

if __name__ == '__main__':

    start_time_all = time.time()

    # set up local arguments to main() function
    date_simulations = "2025-08-20"
    moduleName = "simulation_module.py"
    num_features = 31
    num_params = 2
    CellTypes = ['YFP']
    dt_default = 0.025
    max_cells = 140
    max_time = None
    cell_type_mapping = {'YFP': 0}
    assign_cell_type = True
    use_grandmother_as_parent = False
    find_neighbors = True

    # get stacked mean and std from 1000 simulations for GNN
    parent_dir = "D:/Projects/GNN Research/Data Files/_sim_mean_std_values_gnn/2025-08-20/"  
    final_path = parent_dir + "CVIS_S2_minmaxtest.pkl"  
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
    gamma_min = values["gamma min"] 
    gamma_max = values["gamma max"] 
    reg_param_min = values["reg_param min"] 
    reg_param_max = values["reg_param max"]  

    print("log(gamma) mean: ", gamma_mean)
    print("log(gamma) std: ", gamma_std)
    print("log(reg_param) mean: ", reg_param_mean)
    print("log(reg_param) std: ", reg_param_std)
    ##print("feature mean: ", mean_dict)
    ##print("feature std: ", std_dict)
    print("log(gamma) min: ", gamma_min)
    print("log(gamma) max: ", gamma_max)
    print("log(reg_param) min: ", reg_param_min)
    print("log(reg_param) max: ", reg_param_max)

    saved_model_path = "D:/Projects/GNN Research/Data Files/_model_data_new/CVIS_S2.pth"

    # 31 features
    include_columns = ['ImageNumber', #
                       'ObjectNumber', #
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
    # Step 1: get sim-exp ss
    file_path = "D:/Projects/GNN Research/Data Files/_sim_graph_splitted_data_gnn/2025-08-24_s2"
    data_list = samples_to_sequences(file_path)
    num_exp_samples = 31
    loader = TorchDataLoader(data_list, batch_size=num_exp_samples, shuffle=False, collate_fn=collate_sequences_batch)
    # load the trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models_gatlstmregressor.GAT_LSTM_Regressor(in_dim=num_features,
                                        gnn_hidden_dim=2*num_features,
                                        z_dim=2*num_features,
                                        lstm_hidden_dim=2*num_features,
                                        out_dim=num_params).to(device)
    model.load_state_dict(torch.load(saved_model_path))
    model.eval()

    # predict gamma and reg_param
    with torch.no_grad():
        for flat_batch, seq_ids, t_steps, y_seq in loader:
            flat_batch = flat_batch.to(device)
            seq_ids = seq_ids.to(device)
            t_steps = t_steps.to(device)
            y_seq = y_seq.to(device)  # [B, 2]

            preds, lengths = model(flat_batch, seq_ids, t_steps)  # preds: [B, 2]
            print("targets: ", y_seq)
            print("predictions: ", preds)
            print("num of seq sub-graphs in each sample: ", lengths)
            print()

    gamma_list = [round(float(param[0] * gamma_std + gamma_mean), 5) for param in preds.tolist()]
    alpha_list = [round(float(param[1] * reg_param_std + reg_param_mean), 5) for param in preds.tolist()]
    exp_summary_stats = {}  ### gamma and alpha predictions are already de-normalized here by still in log10 space
    p1 = statistics.mean(gamma_list)  ### take mean over 31 samples
    p2 = statistics.mean(alpha_list)  ### take mean over 31 samples
    exp_summary_stats["gamma"] = (p1-gamma_min)/(gamma_max-gamma_min)  ### min-max normalization
    exp_summary_stats["alpha"] = (p2-reg_param_min)/(reg_param_max-reg_param_min)  ### min-max normalization


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
    n_cores = 2
    population_size = 100 
    min_epsilon = 0.07071
    max_populations = 100

    # Step x: create abc model
    abc = pyabc.ABCSMC(model_test, 
                       prior, 
                       distance_calculation, 
                       population_size=population_size, 
                       sampler=pyabc.sampler.SingleCoreSampler()    ### will cause error if set it to multiplecoresampler for some reasons
                       )

    # Step x: create database only
    #"""
    db_path = "CVIS_S2_minmaxtest.db"
    history = abc.new("sqlite:///D:/Projects/GNN Research/Data Files/abc_results/" + db_path, exp_summary_stats)
    print("ABC-SMC run ID:", history.id)
    #"""

    # to resume a stored run only
    """
    db_path = "CVIS_S2_minmaxtest.db"
    history = abc.load("sqlite:///D:/Projects/GNN Research/Data Files/abc_results/" + db_path, 1)  ## pick up where we were stopped, need to check id using jupyter
    print("num of completed populations: ", history.n_populations)
    print("ABC-SMC run ID: ", history.id)
    ##breakpoint()
    """

    # Step x: run
    history = abc.run(minimum_epsilon=min_epsilon, 
                      max_nr_populations=max_populations) 

    end_time_all = time.time()  ### about 20 hours for population_size*num_populations = 500
    print(f"Training time: {(end_time_all - start_time_all)/3600:.2f} hours")