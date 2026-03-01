from itertools import combinations
from torch_geometric.loader import DataLoader
from torch.utils.data import DataLoader as TorchDataLoader
import matplotlib.pyplot as plt
import pickle as pk
import torch
import torch.nn.functional as F
import os
import sys
import random
from collections import defaultdict
from sklearn.model_selection import train_test_split
import numpy as np
from torch_geometric.data import Data, Dataset

import time
from typing import List
from torch_geometric.data import Batch

sys.path.append('/work/x5bai/project/Code_Files/s4')
import models_gatlstmlearner

def generate_triplets(data):
    """
    Think of triplets as sample indices of data_list.
    data: {'g_x_rp_x':['g_x_rp_x_iter_0','g_x_rp_x_iter_1',...,''], '':['',...,''],..., '':['',...,'']}
    triplets: [('g_x_rp_x_iter_a','g_x_rp_x_iter_p','g_w_rp_w_iter_n'),('','',''),('','',''),...,('','','')]
    data_list: [(([Data1,Data2,...],y),(),()),((),(),()),...,((),(),())]
    """
    triplets = []
    params = list(data.keys())  # List of parameter sets

    for param, instances in data.items(): ## 100 in total
        # Anchor and Positive from the same parameter set
        for anchor, positive in combinations(instances, 2):  ## 45 in total for each param
            # Negative from a different parameter set
            negative_param = random.choice([p for p in params if p != param])  ## randomly choose 1 from 99
            negative = random.choice(data[negative_param])  ## randomly choose 1 from 10
            triplets.append((anchor, positive, negative))
        # print(len([c for c in combinations(instances, 2)]))  ## this is a way to check it's 45 for each param

    return triplets


def generate_data_splits(train_split, test_split, val_split, triplets, data_tuple):

    if train_split + test_split + val_split == 1:

        # processing of triplet file names
        triplets_n = []
        for i in triplets:
            temp = []
            for j in i:
                temp.append(j.split('-')[0])
            triplets_n.append(tuple(temp))

        # Step 1: Create a mapping of anchor-negative pairs
        anchor_negative_map = defaultdict(list)
        k = 0
        for anchor, positive, negative in triplets_n:
            anchor_negative_map[(anchor, negative)].append((k, anchor, positive, negative))
            k += 1

        # Step 2: Get anchor-negative pairs
        anchor_negative_pairs = list(anchor_negative_map.keys())

        # Step 3: Stratified split for training, validation, and test
        train_pairs, temp_pairs = train_test_split(
            anchor_negative_pairs, test_size=1 - train_split, random_state=42
        )
        val_pairs, test_pairs = train_test_split(
            temp_pairs, test_size=val_split / (1 - train_split), random_state=42
        )

        # Step 4: Collect triplets corresponding to each split
        train_triplets = [triplet for pair in train_pairs for triplet in anchor_negative_map[pair]]
        val_triplets = [triplet for pair in val_pairs for triplet in anchor_negative_map[pair]]
        test_triplets = [triplet for pair in test_pairs for triplet in anchor_negative_map[pair]]

        train_ix = list(map(lambda x: x[0], train_triplets))
        val_ix = list(map(lambda x: x[0], val_triplets))
        test_ix = list(map(lambda x: x[0], test_triplets))

        triplets = np.array(triplets)
        train_dat = triplets[train_ix]
        test_dat = triplets[test_ix]
        val_dat = triplets[val_ix]

        final_train_data = []
        final_test_data = []
        final_val_data = []

        for i in train_dat:
            k = []
            for j in i:
                k.append(data_tuple[j])
            final_train_data.append(k)

        for i in test_dat:
            k = []
            for j in i:
                k.append(data_tuple[j])
            final_test_data.append(k)

        for i in val_dat:
            k = []
            for j in i:
                k.append(data_tuple[j])
            final_val_data.append(k)

        return final_train_data, final_test_data, final_val_data


def samples_to_sequences(triplets, file_path):
    """
    input:
        triplets = [('gamma_x_reg_param_x_iter_1','',''),...,('','','')]
        file_path = "D:/Projects/GNN Research/Data Files/_sim_graph_splitted_data_gnn/2025-09-25"
    output:
        data_list = [(([Data1,...,Data5],y),(),()),...,((),(),())]
    note: data_list has the same format as triplets, so we can think triplets as the id of data_list
    """

    folders = [os.path.join(file_path, f) for f in os.listdir(file_path) if
                os.path.isdir(os.path.join(file_path, f))]  ### .../iteration 000/
    data_dict = {}
    for sample_path in folders:
        l1_folder = [os.path.join(sample_path, f) for f in os.listdir(sample_path) if
                    os.path.isdir(os.path.join(sample_path, f))]  ### .../gamma=37.6_reg_param=1.24_iter=0/
        l2_folder = [os.path.join(l1_folder[0], f) for f in os.listdir(l1_folder[0]) if
                    os.path.isdir(os.path.join(l1_folder[0], f))]  ### .../node_feature_data/
        sample_name = [f for f in os.listdir(sample_path)
                       if os.path.isdir(os.path.join(sample_path, f))][0].replace("=","_")
        pkl_file_path = [os.path.join(l2_folder[0], f) for f in os.listdir(l2_folder[0])][0]  ### .../test0826_1000sims_stdnorm_log10_27f.pkl
        with open(pkl_file_path, "rb") as file:
            seq_dict = pk.load(file)  ### now the pickle is loaded

        y_tensor = None
        data = []
        indices = []
        for k,v in seq_dict.items():  ### now need to match it to the same structure as train_ds in test2()
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
        #data_list.append(data_tuple)  ### [Sample1, Sample2, ..., Sample10] across all 1000 samples
        data_dict[sample_name] = data_tuple

    data_list = []
    for t in triplets:
        a,p,n = t
        g_a = data_dict[a]
        g_p = data_dict[p]
        g_n = data_dict[n]
        data_list.append((g_a, g_p, g_n))

    return data_list, data_dict


def split_data(train_split, val_split, test_split, dataset):

    random.shuffle(dataset)

    train_size = int(len(dataset) * train_split)
    test_size = int(len(dataset) * test_split)
    val_size = int(len(dataset) * val_split)

    final_train_data = dataset[:train_size]
    final_test_data = dataset[train_size:train_size + test_size]
    final_val_data = dataset[train_size + test_size:]

    return final_train_data, final_test_data, final_val_data


def collate_sequences_batch_triplets(batch_triplets):
    if type(batch_triplets) == tuple:
        batch = [batch_triplets]

    batch_a = []
    batch_p = []
    batch_n = []
    for t in batch_triplets:
        seq_a, seq_p, seq_n = t
        batch_a.append(seq_a)
        batch_p.append(seq_p)
        batch_n.append(seq_n)

    flat_batch_a, seq_ids_a, t_steps_a, y_seq_a = collate_sequences_batch(batch_a)
    flat_batch_p, seq_ids_p, t_steps_p, y_seq_p = collate_sequences_batch(batch_p)
    flat_batch_n, seq_ids_n, t_steps_n, y_seq_n = collate_sequences_batch(batch_n)

    return ((flat_batch_a, seq_ids_a, t_steps_a, y_seq_a),
            (flat_batch_p, seq_ids_p, t_steps_p, y_seq_p),
            (flat_batch_n, seq_ids_n, t_steps_n, y_seq_n))

def collate_sequences_batch(batch):

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


def train_epoch(loader, model, optimizer, device, criterion):

    model.train()

    total_loss = 0.0
    correct_triplets = 0
    total_triplets = 0

    targets = []
    predictions = []
    margin = 1

    # set things to device
    for loader_a, loader_p, loader_n in loader:  ### loop over batches

        flat_batch_a, seq_ids_a, t_steps_a, y_seq_a = loader_a
        flat_batch_a = flat_batch_a.to(device)
        seq_ids_a = seq_ids_a.to(device)
        t_steps_a = t_steps_a.to(device)
        y_seq_a = y_seq_a.to(device)  # [B, 2]

        flat_batch_p, seq_ids_p, t_steps_p, y_seq_p = loader_p
        flat_batch_p = flat_batch_p.to(device)
        seq_ids_p = seq_ids_p.to(device)
        t_steps_p = t_steps_p.to(device)
        y_seq_p = y_seq_p.to(device)  # [B, 2]

        flat_batch_n, seq_ids_n, t_steps_n, y_seq_n = loader_n
        flat_batch_n = flat_batch_n.to(device)
        seq_ids_n = seq_ids_n.to(device)
        t_steps_n = t_steps_n.to(device)
        y_seq_n = y_seq_n.to(device)  # [B, 2]

        preds_a, lengths_a = model(flat_batch_a, seq_ids_a, t_steps_a)
        preds_p, lengths_p = model(flat_batch_p, seq_ids_p, t_steps_p)
        preds_n, lengths_n = model(flat_batch_n, seq_ids_n, t_steps_n)

        loss = criterion(preds_a, preds_p, preds_n)
        optimizer.zero_grad(set_to_none=True)  ####################################
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  ###########################################
        optimizer.step()

        total_loss += loss.item()
        distance_positive = F.pairwise_distance(preds_a, preds_p, p=2)
        distance_negative = F.pairwise_distance(preds_a, preds_n, p=2)
        correct_triplets += (distance_positive + margin < distance_negative).sum().item()
        total_triplets += len(distance_positive)

        targets.append((y_seq_a, y_seq_p, y_seq_n))
        predictions.append((preds_a, preds_p, preds_n))

        ##print("triplet targets: ", (y_seq_a, y_seq_p, y_seq_n))
        ##print("embedding generations: ", (preds_a, preds_p, preds_n))
        ##print("num of seq sub-graphs in each triplet sample: ", (lengths_a, lengths_p, lengths_n))

    avg_loss = total_loss / len(loader)
    accuracy = (correct_triplets / total_triplets) * 100

    return avg_loss, accuracy, targets, predictions



def val_epoch(loader, model, device, criterion):

    model.eval()

    total_loss = 0.0
    correct_triplets = 0
    total_triplets = 0

    targets = []
    predictions = []
    margin = 1

    with torch.no_grad():
        for loader_a, loader_p, loader_n in loader:  ### loop over batches

            flat_batch_a, seq_ids_a, t_steps_a, y_seq_a = loader_a
            flat_batch_a = flat_batch_a.to(device)
            seq_ids_a = seq_ids_a.to(device)
            t_steps_a = t_steps_a.to(device)
            y_seq_a = y_seq_a.to(device)  # [B, 2]

            flat_batch_p, seq_ids_p, t_steps_p, y_seq_p = loader_p
            flat_batch_p = flat_batch_p.to(device)
            seq_ids_p = seq_ids_p.to(device)
            t_steps_p = t_steps_p.to(device)
            y_seq_p = y_seq_p.to(device)  # [B, 2]

            flat_batch_n, seq_ids_n, t_steps_n, y_seq_n = loader_n
            flat_batch_n = flat_batch_n.to(device)
            seq_ids_n = seq_ids_n.to(device)
            t_steps_n = t_steps_n.to(device)
            y_seq_n = y_seq_n.to(device)  # [B, 2]

            preds_a, lengths_a = model(flat_batch_a, seq_ids_a, t_steps_a)
            preds_p, lengths_p = model(flat_batch_p, seq_ids_p, t_steps_p)
            preds_n, lengths_n = model(flat_batch_n, seq_ids_n, t_steps_n)

            loss = criterion(preds_a, preds_p, preds_n)

            total_loss += loss.item()
            distance_positive = F.pairwise_distance(preds_a, preds_p, p=2)
            distance_negative = F.pairwise_distance(preds_a, preds_n, p=2)
            correct_triplets += (distance_positive + margin < distance_negative).sum().item()
            total_triplets += len(distance_positive)

            targets.append((y_seq_a, y_seq_p, y_seq_n))
            predictions.append((preds_a, preds_p, preds_n))

            ##print("triplet targets: ", (y_seq_a, y_seq_p, y_seq_n))
            ##print("embedding generations: ", (preds_a, preds_p, preds_n))
            ##print("num of seq sub-graphs in each triplet sample: ", (lengths_a, lengths_p, lengths_n))

    avg_loss = total_loss / len(loader)
    accuracy = (correct_triplets / total_triplets) * 100

    return avg_loss, accuracy, targets, predictions



def test_epoch(loader, model, device, criterion):

    model.eval()

    total_loss = 0.0
    correct_triplets = 0
    total_triplets = 0

    targets = []
    predictions = []
    margin = 1

    with torch.no_grad():
        for loader_a, loader_p, loader_n in loader:  ### loop over batches

            flat_batch_a, seq_ids_a, t_steps_a, y_seq_a = loader_a
            flat_batch_a = flat_batch_a.to(device)
            seq_ids_a = seq_ids_a.to(device)
            t_steps_a = t_steps_a.to(device)
            y_seq_a = y_seq_a.to(device)  # [B, 2]

            flat_batch_p, seq_ids_p, t_steps_p, y_seq_p = loader_p
            flat_batch_p = flat_batch_p.to(device)
            seq_ids_p = seq_ids_p.to(device)
            t_steps_p = t_steps_p.to(device)
            y_seq_p = y_seq_p.to(device)  # [B, 2]

            flat_batch_n, seq_ids_n, t_steps_n, y_seq_n = loader_n
            flat_batch_n = flat_batch_n.to(device)
            seq_ids_n = seq_ids_n.to(device)
            t_steps_n = t_steps_n.to(device)
            y_seq_n = y_seq_n.to(device)  # [B, 2]

            preds_a, lengths_a = model(flat_batch_a, seq_ids_a, t_steps_a)
            preds_p, lengths_p = model(flat_batch_p, seq_ids_p, t_steps_p)
            preds_n, lengths_n = model(flat_batch_n, seq_ids_n, t_steps_n)

            loss = criterion(preds_a, preds_p, preds_n)

            total_loss += loss.item()
            distance_positive = F.pairwise_distance(preds_a, preds_p, p=2)
            distance_negative = F.pairwise_distance(preds_a, preds_n, p=2)
            correct_triplets += (distance_positive + margin < distance_negative).sum().item()
            total_triplets += len(distance_positive)

            targets.append((y_seq_a, y_seq_p, y_seq_n))
            predictions.append((preds_a, preds_p, preds_n))

            ##print("triplet targets: ", (y_seq_a, y_seq_p, y_seq_n))
            ##print("embedding generations: ", (preds_a, preds_p, preds_n))
            ##print("num of seq sub-graphs in each triplet sample: ", (lengths_a, lengths_p, lengths_n))

    avg_loss = total_loss / len(loader)
    accuracy = (correct_triplets / total_triplets) * 100

    return avg_loss, accuracy, targets, predictions

def model_save(model, path) -> None:

    torch.save(model.state_dict(), path)

    print("Model saved.")


if __name__ == "__main__":

    # generate triplets
    dt_sim = "2025-09-25"
    parent_dir = f"/work/x5bai/project/Data_Files/_sim_tr_csv_data_gnn/{dt_sim}/"
    iterations = [os.path.join(parent_dir, f) for f in os.listdir(parent_dir) if
                  os.path.isdir(os.path.join(parent_dir, f))]
    data = {}
    records = []
    for i in iterations:
        sub_folder_name = [f for f in os.listdir(i) if os.path.isdir(os.path.join(i, f))][0].replace("=","_")
        if sub_folder_name.split("_iter_")[0] not in records:
            data[sub_folder_name.split("_iter_")[0]] = [sub_folder_name]
            records.append(sub_folder_name.split("_iter_")[0])
        else:
            data[sub_folder_name.split("_iter_")[0]] += [sub_folder_name]
    triplets = generate_triplets(data)
    print("num of triplets: ", len(triplets))

    # generate triplet sequences
    file_path = f"/work/x5bai/project/Data_Files/_sim_graph_splitted_data_gnn/{dt_sim}"
    data_list,data_dict = samples_to_sequences(triplets, file_path)

    # split data
    train_split, val_split, test_split = 0.7, 0.15, 0.15
    train_ds, test_ds, val_ds = split_data(train_split, val_split, test_split, data_list)

    # set up a device for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ##print("train_ds[0]: ",train_ds[0])

    # set up data loaders in batches
    b_size = 25

    train_loader = TorchDataLoader(
        train_ds, batch_size=b_size, shuffle=True,
        collate_fn=collate_sequences_batch_triplets
    )
    val_loader = TorchDataLoader(
        val_ds, batch_size=b_size, shuffle=False,
        collate_fn=collate_sequences_batch_triplets
    )
    te_loader = TorchDataLoader(
        test_ds, batch_size=b_size, shuffle=False,
        collate_fn=collate_sequences_batch_triplets
    )

    # print out the batches for a quick check
    ##b_num = round(len(data_list) / b_size)
    ##for i, batch in zip(range(b_num), train_loader):
    ##    print(f"batch {i + 1}: ", batch)

    # model setup
    num_features = 31
    embedding_size = 6
    model = models_gatlstmlearner.Simple_GAT_LSTM_Learner(in_dim=num_features,
                                                   gnn_hidden_dim=2*num_features,
                                                   z_dim=2*num_features,
                                                   lstm_hidden_dim=2*num_features,
                                                   out_dim=embedding_size).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    mixed_loss_alpha = 0.5
    mixed_loss_margin = 1
    mixed_loss_cosine = 0.2
    criterion = models_gatlstmlearner.MixedTripletLoss(alpha=mixed_loss_alpha,
                                                       margin=mixed_loss_margin,
                                                       margin_cosine=mixed_loss_cosine).to(device)
    print(model)

    # model training and validation
    start_time = time.time()
    best_val_loss = float('inf')
    best_epoch = 0
    counter = 0
    best_model_state = None

    epoch_num_list = []
    train_loss_list = []
    val_loss_list = []

    epochs = 500
    patience = 20
    
    for epoch in range(1, epochs + 1):

        tr_loss, tr_accuracy, tr_targets, tr_predictions = train_epoch(train_loader, model, optimizer, device, criterion)
        va_loss, va_accuracy, va_targets, va_predictions = val_epoch(val_loader, model, device, criterion)

        print(f"Epoch {epoch:02d} | "
              f"train loss: {tr_loss:.4f}, train accuracy: {tr_accuracy:.4f} | "
              f"val loss: {va_loss:.4f}, val accuracy: {va_accuracy:.4f}")

        epoch_num_list.append(epoch)
        train_loss_list.append(tr_loss)
        val_loss_list.append(va_loss)

        # Early stopping logic
        if va_loss < best_val_loss:
            best_val_loss = va_loss
            best_epoch = epoch
            counter = 0
            best_model_state = model.state_dict()  # Save the best model
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}. Best epoch: {best_epoch}")
                break

    # Load the best model before testing
    model.load_state_dict(best_model_state)

    # check the training duration
    end_time = time.time()
    print(f"Training time: {(end_time - start_time) / 3600:.2f} hrs")

    # plots
    x = epoch_num_list
    y1 = train_loss_list
    y2 = val_loss_list

    plt.plot(x, y1, label='Training loss', linestyle='-', marker='o')
    plt.plot(x, y2, label='Validation loss', linestyle='--', marker='s')

    plt.xlabel('Epoch Number')
    plt.ylabel('Loss')
    plt.title('Training/Validation Loss vs Epoch Number')
    plt.legend()
    plt.show()

    # test
    te_loss, te_accuracy, te_targets, te_predictions = test_epoch(te_loader, model, device, criterion)
    print(f"test loss: {te_loss:.4f}, test accuracy: {te_accuracy:.4f}")

    # save
    save_path = "/work/x5bai/project/Data_Files/_model_data_old/CVIS_S4_minmaxtest.pth"
    model_save(model, save_path)
    
    # get min max values of ss from training data
    data_list = list(data_dict.values())
    loader = TorchDataLoader(data_list, batch_size=1, shuffle=False, collate_fn=collate_sequences_batch)

    load_path = "/work/x5bai/project/Data_Files/_model_data_old/CVIS_S4_minmaxtest.pth"
    model.load_state_dict(torch.load(load_path))
    model.eval()

    with torch.no_grad():  ### predict gamma and reg_param
        for flat_batch, seq_ids, t_steps, y_seq in loader:  ### one batch here and batch size = 31
            flat_batch = flat_batch.to(device)
            seq_ids = seq_ids.to(device)
            t_steps = t_steps.to(device)
            y_seq = y_seq.to(device)  # [B, 2]

            preds, lengths = model(flat_batch, seq_ids, t_steps)  # preds: [B, 2]

    e1_list = [round(float(param[0]), 5) for param in preds.tolist()]
    e2_list = [round(float(param[1]), 5) for param in preds.tolist()]
    e3_list = [round(float(param[2]), 5) for param in preds.tolist()]
    e4_list = [round(float(param[3]), 5) for param in preds.tolist()]
    e5_list = [round(float(param[4]), 5) for param in preds.tolist()]
    e6_list = [round(float(param[5]), 5) for param in preds.tolist()]

    min_ss1 = np.min(e1_list)
    min_ss2 = np.min(e2_list)
    min_ss3 = np.min(e3_list)
    min_ss4 = np.min(e4_list)
    min_ss5 = np.min(e5_list)
    min_ss6 = np.min(e6_list)
    max_ss1 = np.max(e1_list)
    max_ss2 = np.max(e2_list)
    max_ss3 = np.max(e3_list)
    max_ss4 = np.max(e4_list)
    max_ss5 = np.max(e5_list)
    max_ss6 = np.max(e6_list)
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

    values = {"min ss1": min_ss1,
              "min ss2": min_ss2,
              "min ss3": min_ss3,
              "min ss4": min_ss4,
              "min ss5": min_ss5,
              "min ss6": min_ss6,
              "max ss1": max_ss1,
              "max ss2": max_ss2,
              "max ss3": max_ss3,
              "max ss4": max_ss4,
              "max ss5": max_ss5,
              "max ss6": max_ss6,
              }

    folder_path = f"/work/x5bai/project/Data_Files/_sim_mean_std_values_gnn/{dt_sim}/"
    os.makedirs(folder_path, exist_ok=True)
    final_path = os.path.join(folder_path, "CVIS_S4_minmaxss.pkl")
    with open(final_path, 'wb') as f:
        pk.dump(values, f)

    print("this number should equal the total number of simulated data: ", len(e1_list))


