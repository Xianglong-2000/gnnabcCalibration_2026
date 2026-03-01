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


sys.path.append('C:/Users/MECHREV/CellModeller-ingallslab')
import output_processing.CellModellerProcessing

sys.path.append('D:/Projects/GNN Research/Code Files/my_code_files/general')
import csv2pre_tr_csv

sys.path.append('D:/Projects/GNN Research/Code Files/my_code_files/s1')
import tr_csv2g_data_gatregressor


def model(parameters:dict): 

    """
    input: log10 gamma and log10 alpha
    go through things in order: pickle -> csv -> pre_tr_csv -> tr_csv -> graph_data
    """

    # set up parameters back to regular space for simulations
    gamma = round(10**parameters["gamma"], 5)
    reg_param = round(10**parameters["reg_param"], 5)
    print(f"parameters for the current simulation: gamma = {gamma}, alpha = {reg_param}")

    # defining input/output locations
    export_path = f"D:/Projects/GNN Research/Data Files/_simulator_pkl_data_abc/{date_simulations}/gamma={gamma}_reg_param={reg_param}"
    os.makedirs(export_path, exist_ok=True)
    ##sys.stdout = open(os.devnull, 'w')  # Disable printing from simulations

    # get pickles
    print("simulation started")
    start_time_each_simulation = time.time()

    # this is to avoid the unexpected simulation error for at most 2 times
    try:
        simulate(moduleName, reg_param=reg_param, gamma=gamma, max_cells = max_cells, export_path = export_path)
    except Exception as e:
        print("Error in simulation: ", e)
        try:
            simulate(moduleName, reg_param=reg_param, gamma=gamma, max_cells=max_cells, export_path=export_path)
        except Exception as e2:
            print("Error in simulation again: ", e2)
            simulate(moduleName, reg_param=reg_param, gamma=gamma, max_cells=max_cells, export_path=export_path)
            os.makedirs(f"D:/Projects/GNN Research/Data Files/_simulator_pkl_data_abc/sim_error_counter/sim_error2_{gamma}_{reg_param}", exist_ok=True)
        else:
            os.makedirs(f"D:/Projects/GNN Research/Data Files/_simulator_pkl_data_abc/sim_error_counter/sim_error1_{gamma}_{reg_param}", exist_ok=True)

    end_time_each_simulation = time.time()


    if end_time_each_simulation - start_time_each_simulation >= 1200:
        if os.path.exists(export_path):
            os.remove(export_path)
            print(f"File '{export_path}' deleted successfully.")
        else:
            print(f"File '{export_path}' does not exist.")
    ##sys.stdout = sys.__stdout__  # Re-enable printing
    print("Simulation completed")
    
    # pickle to csv
    print("pickle to csv started")
    paths = [os.path.join(export_path, f) for f in os.listdir(export_path) if os.path.isdir(os.path.join(export_path, f))]
    input_dir = paths[0]
    #sub_paths = [os.path.join(paths[0], f) for f in os.listdir(paths[0])]
    #input_dir = sub_paths
    #print(input_dir)
    output_dir = f"D:/Projects/GNN Research/Data Files/_simulator_csv_data_abc/{date_simulations}/gamma={gamma}_reg_param={reg_param}"
    os.makedirs(output_dir, exist_ok=True)
    try:
        output_processing.CellModellerProcessing.process_simulation_directory(
                    input_directory=input_dir,
                    cell_type_mapping=cell_type_mapping,
                    output_directory=output_dir,
                    assign_cell_type=assign_cell_type,
                    use_grandmother_as_parent=use_grandmother_as_parent,
                    find_neighbors=find_neighbors)
    except Exception as e:
        print("Error in outputprocessing: ", e)
        os.makedirs(f"D:/Projects/GNN Research/Data Files/_simulator_pkl_data_abc/sim_error_counter/pkl2csv_error_{gamma}_{reg_param}", exist_ok=True)
        model(parameters)
    print("pickle to csv completed")

    # csv to pre_tr_csv
    input_dir = output_dir
    output_dir = f"D:/Projects/GNN Research/Data Files/_simulator_pre_tr_csv_data_abc/{date_simulations}/gamma={gamma}_reg_param={reg_param}"
    os.makedirs(output_dir, exist_ok=True)
    csv2pre_tr_csv.delete_columns(input_dir + "/Objects properties.csv", output_dir + "/Objects properties.csv")
    shutil.copy(input_dir + "/Object relationships.csv", output_dir + "/")

    # pre_tr_csv to tr_csv
    print("pre_tr csv to tr csv started")
    input_dir = output_dir
    output_dir = Path(f"D:/Projects/GNN Research/Data Files/_simulator_tr_csv_data_abc/{date_simulations}/gamma={gamma}_reg_param={reg_param}")
    os.makedirs(output_dir, exist_ok=True)
    shutil.copy(input_dir + "/Object relationships.csv", str(output_dir) + "/")

    # Start Processing
    # Simulate running the CLI with valid arguments
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"

    result = subprocess.run(
        ["D:/Projects/GNN Research/Code Files/myenv/Scripts/python.exe",
         "-m", "Trackrefiner.cli",
         "-i", input_dir + "/Objects properties.csv",
         "-n", input_dir + "/Object relationships.csv",
         "-t", "3",
         "-d", "20",
         "-p", "0.14",
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
    ## print("STDOUT:", repr(result.stdout))
    ## print("STDERR:", repr(result.stderr))
    try:
        assert result.returncode == 0  # Ensure the process exits successfully
    except Exception as e:
        print("Error in TrackRefiner: ", e)
        os.makedirs(f"D:/Projects/GNN Research/Data Files/_simulator_pkl_data_abc/sim_error_counter/pre_tr_csv2tr_csv_error_{gamma}_{reg_param}", exist_ok=True)
        model(parameters)
    # Check for the final success log message in stdout
    assert "Trackrefiner Process completed at:" in result.stdout, ("Expected log message indicating successful completion was not found in stdout.")
    assert output_dir.exists()
    print("pre_tr csv to tr csv completed")

    # tr_csv to graph
    # adapt this part to the gatlstm strategy
    print("tr csv to graphs started")
    graph_id = f"gamma={gamma}_reg_param={reg_param}"
    csv1_path = str(output_dir)+"/Trackrefiner.Objects properties_Average_analysis.csv"
    csv2_path = str(output_dir)+"/Object relationships.csv"
    data = load_graph_data(graph_id, csv1_path, csv2_path, include_columns)
    folder_path = "D:/Projects/GNN Research/Data Files/_simulator_graph_data_abc/2025-08-20/"
    os.makedirs(folder_path, exist_ok=True)
    g_data_path = os.path.join(folder_path, f"abc_g_data_{graph_id}.pkl")
    with open(g_data_path, 'wb') as f:
        pk.dump(data, f)
    print("tr csv to graphs completed")

def simulate(modfilename, reg_param, gamma, max_cells=None, max_time=None, export_path=None):

    (path,name) = os.path.split(modfilename)
    modname = str(name).split('.')[0]
    sys.path.append(path)

    output_dir = export_path  
    ##print(output_dir)

    os.makedirs(output_dir, exist_ok=True)  # create a new folder if it doesn't exist and do nothing if it exists
    os.chdir(output_dir)  # generate the simulated data in that folder

    # Extract dt and sim_time from the simulation module
    sim_file = "D:/Projects/GNN Research/Code Files/my_code_files/general/"+modfilename
    (dt, sim_time) = sim_time_parameters(sim_file)

    params = {"gamma": gamma, "reg_param": reg_param}
    print("check parameters right before the current simulation: ", params.values())
    sim = Simulator(modname, dt, pickleSteps=2, saveOutput = True, psweep=True, params=params)

    if max_cells:
        while len(sim.cellStates) <= max_cells:
            sim.step()


def sim_time_parameters(file_path:str):
    """
    Extract the dt and sim_time from the module file. 
    Must be written as dt = X.Y and sim_time = A.B in the module file.
    The argument file_path is the name of the simulation module file.
    """
    
    search_dt = "dt = (\d+\.\d+)"
    search_sim_time = "sim_time = (\d+\.\d+)"
    with open(file_path, "r+") as file:
        file_contents = file.read()
        dt = re.findall(search_dt, file_contents)
        sim_time = re.findall(search_sim_time, file_contents)
        if len(dt) == 0:
            dt_result = float(dt_default)
        else:
            dt_result = float(dt[0])
        if len(sim_time) == 0:
            sim_time_result = max_time
        else:
            sim_time_result = float(sim_time[0])

    return dt_result, sim_time_result

def load_graph_data(graph_ID:str, node_features_file:str, neighbours_edge_file:str, include_columns:list):
    
    # Create the y vector, y = [gamma, reg_param]
    print("the current simulation is ", graph_ID)
    y1_original = float(graph_ID.split('_')[0].split('=')[-1])
    y2_original = float(graph_ID.split('_')[-1].split('=')[-1])
    y1 = (math.log10(y1_original) - gamma_mean) / gamma_std   ### standard normalization
    y2 = (math.log10(y2_original) - reg_param_mean) / reg_param_std
    y = torch.tensor([[y1],
                      [y2]], dtype=torch.float)
    
    # Read though the csv files and create the df
    node_features_df = pd.read_csv(node_features_file, dtype=str)
    neighbours_edge_df = pd.read_csv(neighbours_edge_file, dtype=str)

    parents_edge_df = tr_csv2g_data_gatregressor.generate_parent_df(node_features_df)
    edge_features_df = pd.concat([parents_edge_df, neighbours_edge_df])

    for col in node_features_df.columns:
        node_features_df[col] = node_features_df[col].map(tr_csv2g_data_gatregressor.smart_cast)

    for col in edge_features_df.columns:
        edge_features_df[col] = edge_features_df[col].map(tr_csv2g_data_gatregressor.smart_cast)

    node_features_df['dir'] = node_features_df['dir'].apply(eval).apply(lambda x: tuple(map(float, x)))
    vec_df = pd.DataFrame(node_features_df['dir'].tolist(), columns=['dir_1', 'dir_2'])
    node_features_df = pd.concat([node_features_df.drop(columns=['dir']), vec_df], axis=1)

    # print(len(include_columns))
    node_features_df = node_features_df[include_columns]
    node_features_df = node_features_df.fillna(node_features_df.mean(numeric_only=True))

    node_index_df = node_features_df.iloc[:, :4]
    node_features_df = node_features_df.iloc[:, 4:]

    df_normalized = node_features_df.copy()

    # Standard normalization
    binary_columns = ["divideFlag", "Unexpected_End", "Unexpected_Beginning"]
    for col in node_features_df.columns:
        if col not in binary_columns:
            df_normalized[col] = (node_features_df[col] - mean_dict[col])/std_dict[col]
            ##print(mean_dict[col], std_dict[col])
    ##print(df_normalized["divideFlag"])
    ##print(df_normalized.head())

    node_features_df = df_normalized

    # print(node_features_df.dtypes)
    # print(node_features_df.head(3))
    # print(len(node_features_df.columns))
    # print(node_features_df.corr().to_string())

    # Create the graph
    G = nx.MultiDiGraph()

    # Add nodes with their features as attributes and label them by their index
    for idx, row in node_features_df.iterrows():
        node_id = idx  # Use the index as a unique label for each node
        G.add_node(node_id, **row.to_dict())

    # Create a unique mapping from (stepNum, ObjectNum) to node index
    node_mapping = {(row['ImageNumber'], row['ObjectNumber']): idx for idx, row in node_index_df.iterrows()}
    contact_edges = []
    
    # Add edges, considering contact edges
    for _, row in edge_features_df.iterrows():
        if row['Relationship'] == 'Neighbors':
            step_num_1 = row['First Image Number']
            step_num_2 = row['Second Image Number']
            node1 = row['First Object Number']
            node2 = row['Second Object Number']

            if (step_num_1, node1) in node_mapping and (step_num_2, node2) in node_mapping:
                if not G.has_edge(node_mapping[(step_num_1, node1)], node_mapping[(step_num_2, node2)]):
                    node1_idx = node_mapping[(step_num_1, node1)]
                    node2_idx = node_mapping[(step_num_2, node2)]
                    G.add_edge(node1_idx, node2_idx, edge_type = 'contact')
                    G.add_edge(node2_idx, node1_idx, edge_type = 'contact')
                    contact_edges.append((node1_idx, node2_idx))
                    contact_edges.append((node2_idx, node1_idx))

    # Create lineage mapping
    lineage_mapping = {(row['id'], row['parent_id'], row['ImageNumber']): idx for idx, row in node_index_df.iterrows()}
    lineage_edges = []
    
    # Add directed lineage edges based on the lineage mapping
    for key, node1_idx in lineage_mapping.items():
        id, parent_id, step_num = key
        if parent_id == 0:
            continue  # Skip if parent_id is zero
        if (id, id, step_num + 1) in lineage_mapping:
            node2_idx = lineage_mapping[(id, id, step_num + 1)]
            G.add_edge(node1_idx, node2_idx, edge_type='lineage')
            lineage_edges.append((node1_idx, node2_idx))
        elif (id, parent_id, step_num + 1) in lineage_mapping:
            node2_idx = lineage_mapping[(id, parent_id, step_num + 1)]
            G.add_edge(node1_idx, node2_idx, edge_type='lineage')
            lineage_edges.append((node1_idx, node2_idx))
        else:
            for parent_key, node2_idx in lineage_mapping.items():
                parent_id_key, _, step_num_key = parent_key
                if parent_id_key == parent_id and step_num_key == step_num - 1:
                    G.add_edge(node2_idx, node1_idx, edge_type='lineage')
                    lineage_edges.append((node2_idx, node1_idx))
                    break

    x = torch.tensor([tr_csv2g_data_gatregressor.flatten_node_attributes(G.nodes[node]) for node in G.nodes()], dtype=torch.float)
    contact_edge_index = torch.tensor(contact_edges, dtype=torch.long).t().contiguous()
    lineage_edge_index = torch.tensor(lineage_edges, dtype=torch.long).t().contiguous()

    # print(x.shape)  # all features in a form of vectors has been spread out as individual columns

    data =  Data(x, contact_edge_index=contact_edge_index, lineage_edge_index=lineage_edge_index, y=y)
    
    return data


if __name__ == '__main__':
    # set up local arguments to main() function
    date_simulations = "2025-08-20"
    moduleName = "simulation_module.py"
    CellTypes = ['YFP']
    dt_default = 0.025
    max_cells = 140
    max_time = None
    cell_type_mapping = {'YFP': 0}
    assign_cell_type = True
    use_grandmother_as_parent = False
    find_neighbors = True

    parent_dir = "D:/Projects/GNN Research/Data Files/_sim_mean_std_values_gnn/2025-08-20/"  ### get stacked mean and std from 1000 simulations for GNN
    final_path = parent_dir + "CVIS_S1.pkl"  ### get stacked mean and std from 1000 simulations for GNN
    print(final_path)  ### get stacked mean and std from 1000 simulations for GNN
    with open(final_path, "rb") as file:  ### get stacked mean and std from 1000 simulations for GNN
        values = pk.load(file)  ### get stacked mean and std from 1000 simulations for GNN
    ##print(values)  ### get stacked mean and std from 1000 simulations for GNN
    gamma_mean = values["gamma mean"]  ### get stacked mean and std from 1000 simulations for GNN
    gamma_std = values["gamma std"]  ### get stacked mean and std from 1000 simulations for GNN
    reg_param_mean = values["reg_param mean"]  ### get stacked mean and std from 1000 simulations for GNN
    reg_param_std = values["reg_param std"]  ### get stacked mean and std from 1000 simulations for GNN
    mean_dict = values["feature mean"]  ### get stacked mean and std from 1000 simulations for GNN
    std_dict = values["feature std"]  ### get stacked mean and std from 1000 simulations for GNN

    print("log(gamma) mean: ", gamma_mean)
    print("log(gamma) std: ", gamma_std)
    print("log(reg_param) mean: ", reg_param_mean)
    print("log(reg_param) std: ", reg_param_std)
    ##print("feature mean: ", mean_dict)
    ##print("feature std: ", std_dict)

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
 
    #gamma = math.log10(250)  ### for a simple test only
    #reg_param = math.log10(30)  ### for a simple test only
    gamma = float(sys.argv[1])
    reg_param = float(sys.argv[2])
    #parameters = sys.argv[1]
    parameters = {"gamma":gamma, "reg_param":reg_param}
    print(parameters)
    model(parameters)
    print()
    
