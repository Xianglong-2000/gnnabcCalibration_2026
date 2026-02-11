import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
import ml_dl_models
import training_new
import pickle as pk
import random
from torch_geometric.loader import DataLoader

@torch.no_grad()
def compute_avg_feature_importance_over_batches(
    model, dataloader,
    contact_layers=['conv1', 'conv2', 'conv3'],
    lineage_layers=['conv3', 'conv4'],
    top_k=25,
    save_csv_path=None,
    columns=None
):
    model.eval()
    device = next(model.parameters()).device

    # Init accumulators
    contact_score_sum = None
    lineage_score_sum = None
    num_batches = 0

    for data in dataloader:
        data = data.to(device)
        x = data.x

        # Run per-edge-type analysis
        contact_scores = feature_scores_from_layers(model, data, contact_layers, 'contact')
        lineage_scores = feature_scores_from_layers(model, data, lineage_layers, 'lineage')

        # Accumulate scores
        if contact_score_sum is None:
            contact_score_sum = contact_scores
            lineage_score_sum = lineage_scores
        else:
            contact_score_sum += contact_scores
            lineage_score_sum += lineage_scores

        num_batches += 1

    # Average over batches
    avg_contact_scores = contact_score_sum / num_batches
    avg_lineage_scores = lineage_score_sum / num_batches

    # Combine edge types: simple average (you could also weight them differently)
    combined_scores = (avg_contact_scores + avg_lineage_scores) / 2
    combined_scores = combined_scores / combined_scores.sum()  # normalize

    topk_idx = torch.topk(combined_scores, top_k).indices
    topk_idx_list = topk_idx.cpu().numpy().tolist()
    topk_names = [columns[i] for i in topk_idx_list]

    print(len(list(range(len(combined_scores)))))
    print(len(columns))
    print(len(combined_scores.cpu().numpy()))

    # Save to CSV
    df = pd.DataFrame({
        'feature_index': list(range(len(combined_scores))),
        'feature_name': columns,
        'importance_score': combined_scores.cpu().numpy()
    })
    df.to_csv(save_csv_path, index=False)
    print(f"Saved feature importance to: {save_csv_path}")

    # Plot top-k
    plt.figure(figsize=(20, 8))
    plt.bar(range(top_k), combined_scores[topk_idx].cpu().numpy())
    plt.xticks(range(top_k), topk_names, rotation=90, fontsize=8)
    plt.xlabel("Feature Index")
    plt.ylabel("Importance Score")
    plt.title(f"Top-{top_k} Features (Combined Contact + Lineage)")
    plt.tight_layout()
    plt.show()

    return combined_scores


@torch.no_grad()
def compute_contact_feature_importance(
    model,
    dataloader,
    contact_layers=['conv1', 'conv2'],
    top_k=25,
    save_csv_path=None,
    columns=None
):
    model.eval()
    device = next(model.parameters()).device

    contact_score_sum = None
    num_batches = 0

    for data in dataloader:
        data = data.to(device)

        # Compute attention-based feature importance for contact edges
        contact_scores = feature_scores_from_layers(model, data, contact_layers, edge_type='contact')

        if contact_score_sum is None:
            contact_score_sum = contact_scores
        else:
            contact_score_sum += contact_scores

        num_batches += 1

    # Average over batches and normalize
    avg_scores = contact_score_sum / num_batches
    avg_scores = avg_scores / avg_scores.sum()  # normalize to sum = 1

    topk_idx = torch.topk(avg_scores, top_k).indices
    topk_idx_list = topk_idx.cpu().numpy().tolist()
    topk_names = [columns[i] for i in topk_idx_list]

    # Save to CSV
    df = pd.DataFrame({
        'feature_index': list(range(len(avg_scores))),
        'feature_name': columns,
        'importance_score': avg_scores.cpu().numpy()
    })
    df.to_csv(save_csv_path, index=False)
    print(f"Saved contact-only feature importance to: {save_csv_path}")

    # Plot top-k
    plt.figure(figsize=(20, 8))
    plt.bar(range(top_k), avg_scores[topk_idx].cpu().numpy())
    plt.xticks(range(top_k), topk_names, rotation=90, fontsize=8)
    plt.xlabel("Feature Index")
    plt.ylabel("Importance Score")
    plt.title(f"Top-{top_k} Most Important Features (Contact Edges Only)")
    plt.tight_layout()
    plt.show()

    return avg_scores


@torch.no_grad()
def compute_lineage_feature_importance(model,
                                       dataloader,
                                       lineage_layers=['conv3', 'conv4', 'conv5'],
                                       top_k=25,
                                       save_csv_path=None,
                                       columns=None):
    model.eval()
    device = next(model.parameters()).device

    lineage_score_sum = None
    num_batches = 0

    for data in dataloader:
        data = data.to(device)

        # Compute attention-based feature importance for lineage edges
        lineage_scores = feature_scores_from_layers(model, data, lineage_layers, edge_type='lineage')

        if lineage_score_sum is None:
            lineage_score_sum = lineage_scores
        else:
            lineage_score_sum += lineage_scores

        num_batches += 1

    # Average over batches and normalize
    avg_scores = lineage_score_sum / num_batches
    avg_scores = avg_scores / avg_scores.sum()  # normalize to sum = 1

    topk_idx = torch.topk(avg_scores, top_k).indices
    topk_idx_list = topk_idx.cpu().numpy().tolist()
    topk_names = [columns[i] for i in topk_idx_list]

    # Save to CSV
    df = pd.DataFrame({
        'feature_index': list(range(len(avg_scores))),
        'feature_name': columns,
        'importance_score': avg_scores.cpu().numpy()
    })
    df.to_csv(save_csv_path, index=False)
    print(f"Saved lineage-only feature importance to: {save_csv_path}")

    # Plot top-k
    plt.figure(figsize=(20, 8))
    plt.bar(range(top_k), avg_scores[topk_idx].cpu().numpy())
    plt.xticks(range(top_k), topk_names, rotation=90, fontsize=8)
    plt.xlabel("Feature Index")
    plt.ylabel("Importance Score")
    plt.title(f"Top-{top_k} Most Important Features (Lineage Edges Only)")
    plt.tight_layout()
    plt.show()

    return avg_scores



@torch.no_grad()
def feature_scores_from_layers(model, data, layers, edge_type):
    x = data.x
    edge_index = getattr(data, f"{edge_type}_edge_index")
    score_sum = torch.zeros(x.size(1), device=x.device)

    for layer_name in layers:
        layer = getattr(model, layer_name)
        _, (e_idx, att_weights) = layer(x, edge_index, return_attention_weights=True)
        att_mean = att_weights.mean(dim=1)
        src, tgt = e_idx
        x_diff = (x[src] - x[tgt]).abs()
        score = x_diff.T @ att_mean
        score_sum += score

    return score_sum


def split_data(dataset: list) -> tuple:
    random.shuffle(dataset)

    train_size = int(len(dataset) * train_split)
    test_size = int(len(dataset) * test_split)
    val_size = int(len(dataset) * val_split)

    final_train_data = dataset[:train_size]
    final_test_data = dataset[train_size:train_size + test_size]
    final_val_data = dataset[train_size + test_size:]

    return final_train_data, final_test_data, final_val_data

if __name__ == '__main__':

    # define features
    include_columns = ['AreaShape_Area',
                       'AreaShape_Center_X',
                       'AreaShape_Center_Y',
                       'AreaShape_MajorAxisLength',
                       'AreaShape_Orientation',
                       'Location_Center_X',
                       'Location_Center_Y',
                       'time',
                       'cellAge',
                       'LifeHistory',
                       'TrajectoryX',
                       'TrajectoryY',
                       'Endpoint1_X',
                       'Endpoint1_Y',
                       'Endpoint2_X',
                       'Endpoint2_Y',
                       'Prev_Center_X',
                       'Prev_Center_Y',
                       'Prev_Endpoint1_X',
                       'Prev_Endpoint1_Y',
                       'Prev_Endpoint2_X',
                       'Prev_Endpoint2_Y',
                       'Bacterium_Slope',
                       'Orientation_Angle_Between_Slopes',
                       'Direction_of_Motion',
                       'Motion_Alignment_Angle',
                       'Source_Neighbor_Avg_TrajectoryX',
                       'Source_Neighbor_Avg_TrajectoryY',
                       'divideFlag',
                       'Division_TimeStep',
                       'Division_Family_Count',
                       'Daughter_Mother_Length_Ratio',
                       'Total_Daughter_Mother_Length_Ratio',
                       'Max_Daughter_Mother_Length_Ratio',
                       'Daughter_Avg_TrajectoryX',
                       'Daughter_Avg_TrajectoryY',
                       'Neighbor_Difference_Count',
                       'Neighbor_Shared_Count',
                       'Average_Length',
                       'Length_Change_Ratio',
                       'Avg_Length_Change_Ratio',
                       'Unexpected_End',
                       'Unexpected_Beginning',
                       'Velocity',
                       'Instant_Velocity',
                       'Average_Instant_Velocity',
                       'Elongation_Rate',
                       'Instant_Elongation_Rate',
                       'strainRate',
                       'strainRate_rolling',
                       'startVol',
                       'targetVol',
                       'Prev_Bacterium_Slope',
                       'Prev_MajorAxisLength',
                       'Bacterium_Movement',
                       'dir_1', 'dir_2']

    # load model
    in_channels_model = 57
    hidden_channels_model = 57
    feature_embedding_size = 2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ml_dl_models.GAT_Regression_Attention_Analysis(in_channels=in_channels_model,
                                        hidden_channels=hidden_channels_model,
                                        out_channels=feature_embedding_size).to(device)
    model.load_state_dict(torch.load("D:/Projects/GNN Research/Data Files/_model_data_new/both_std_log_57f_attention_analysis_2025-08-20.pth"))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # define training loader again
    parent_dir = "D:/Projects/GNN Research/Data Files/_sim_graph_data_gnn/2025-08-20/"
    g_data_path = parent_dir + "test_0820std_log10.pkl"
    with open(g_data_path, "rb") as file:
        data_tuple = pk.load(file)
    train_split, test_split, val_split = 0.7, 0.15, 0.15
    dataset = list(data_tuple.copy().values())
    final_train_data, final_test_data, final_val_data = split_data(dataset)
    batch_train_size = 1
    train_loader = DataLoader(final_train_data, batch_size=batch_train_size, shuffle=True)

    # run the attention score analysis for both edge types
    combined_feature_scores = compute_avg_feature_importance_over_batches(
        model=model,
        dataloader=train_loader,  # or train_loader
        contact_layers=['conv1', 'conv2'],
        lineage_layers=['conv3', 'conv4', 'conv5'],
        top_k=57,
        save_csv_path='D:/Projects/GNN Research/Data Files/_feature_selection/avg_feature_importance_2025-08-20.csv',
        columns=include_columns
    )

    # run the attention score analysis for contact edge type
    contact_feature_scores = compute_contact_feature_importance(
        model=model,
        dataloader=train_loader,  # or train_loader
        contact_layers=['conv1', 'conv2'],
        top_k=57,
        save_csv_path='D:/Projects/GNN Research/Data Files/_feature_selection/contact_feature_importance_2025-08-20.csv',
        columns=include_columns
    )

    # run the attention score analysis for lineage edge type
    lineage_feature_scores = compute_lineage_feature_importance(
        model=model,
        dataloader=train_loader,  # or train_loader
        lineage_layers=['conv3', 'conv4', 'conv5'],
        top_k=57,
        save_csv_path='D:/Projects/GNN Research/Data Files/_feature_selection/lineage_feature_importance_2025-08-20.csv',
        columns=include_columns
    )

