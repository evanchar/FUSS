import numpy as np
import torch
import numpy as nn
import matplotlib.pyplot as plt
import os
from omegaconf import OmegaConf
from sklearn.manifold import TSNE
import seaborn as sns
# from umap import UMAP
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from scipy.optimize import linear_sum_assignment

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# cfg = OmegaConf.load("../src/configs/train_config_cocostuff_MA3.yaml")
cfg = OmegaConf.load("../configs/train_config_cityscapes_MA3.yaml")

# todo : fedavg
root_fedavg = ""

# todo: fedcc
root_fedcc = ""


def maximin_clustering(centroids, num_clusters):
    """
    Perform Maximin clustering to select 'num_clusters' cluster centers.

    Args:
        centroids (torch.Tensor): Tensor of shape (N, D) where N is the number of initial centroids.
        num_clusters (int): Number of clusters to select.

    Returns:
        torch.Tensor: Selected cluster centers of shape (num_clusters, D)
    """
    device = centroids.device
    selected_centroids = [centroids[0]]  # Start with the first centroid

    while len(selected_centroids) < num_clusters:
        # Compute distances from all centroids to the selected centroids
        distances = torch.stack([
            torch.norm(centroids - c, dim=1) for c in selected_centroids
        ])  # Shape: (num_selected, N)

        # Get the minimum distance for each centroid (i.e., its closest selected centroid)
        min_distances = torch.min(distances, dim=0)[0]  # Shape: (N,)

        # Select the centroid that has the maximum minimum distance
        next_centroid_idx = torch.argmax(min_distances).item()
        selected_centroids.append(centroids[next_centroid_idx])

    return torch.stack(selected_centroids, dim=0)  # Shape: (num_clusters, D)


def hungarian(embedding_result, class_labels,
              num_classes, pca_result_maximin):
    ###hungarian matching
    # Step 1: Compute class-wise means from original (PCA-reduced)
    class_means = np.array([
        embedding_result[class_labels == i].mean(axis=0) for i in range(num_classes)
    ])

    # Step 2: Compute cost matrix between maximin centroids and class means
    cost_matrix = pairwise_distances(pca_result_maximin, class_means)

    # Step 3: Hungarian matching (minimum total distance)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Reorder maximin centroids to match class order
    pca_result_maximin_ordered = pca_result_maximin[row_ind[np.argsort(col_ind)]]
    return pca_result_maximin_ordered

def client_selector(root):
    clients = []
    client_num = cfg.client_num
    for i in range(client_num):
        clients.append(torch.load(os.path.join(root, f"clusters_client{i}_agg9")))

    aggs = cfg.aggregation_num
    steps_per_agg = int(len(clients[0]) / aggs)

    dists = np.zeros((client_num, aggs))

    # Set Seaborn theme
    sns.set_context("talk")
    sns.set_style("whitegrid")

    num_classes = 10
    step = steps_per_agg-1

    # Example tensors A and B, shape (27, 70)
    A = clients[0][step][num_classes:num_classes+10].cpu()
    B = clients[1][step][num_classes:num_classes+10].cpu()
    C = clients[2][step][num_classes:num_classes+10].cpu()


    # Concatenate all for t-SNE
    combined = torch.cat([A, B, C], dim=0)
    combine_original = torch.cat([clients[0][step], clients[1][step], clients[2][step]]).cpu()

    if "mm" in root:
        new_centroids = maximin_clustering(combine_original, num_clusters=27)
    else:
        new_centroids = ((clients[0][step] + clients[1][step] + clients[2][step]) / client_num).cpu()

    A_next = clients[0][2 * step + 1][num_classes:num_classes + 10].cpu()
    B_next = clients[1][2 * step + 1][num_classes:num_classes + 10].cpu()
    C_next = clients[2][2 * step + 1][num_classes:num_classes + 10].cpu()

    # Concatenate all for t-SNE
    combined2 = torch.cat([A_next, B_next, C_next], dim=0)
    combine_original2 = torch.cat([clients[0][2 * step + 1], clients[1][2 * step + 1], clients[2][2 * step + 1]]).cpu()

    if "mm" in root:
        new_centroids2 = maximin_clustering(combine_original2, num_clusters=27)
    else:
        new_centroids2 = (
                    (clients[0][2 * step + 1] + clients[1][2 * step + 1] + clients[2][2 * step + 1]) / client_num).cpu()

    # return A,B,C, A_next, B_next, C_next, combined, combined2

    return combined, combined2, new_centroids, new_centroids2, num_classes


############################################################################################################



combined_fedavg, combined2_fedavg, new_centroids_fedavg, new_centroids2_fedavg, num_classes = client_selector(root_fedavg)
combined_fedcc, combined2_fedcc, new_centroids_fedcc, new_centroids2_fedcc, num_fedcc = client_selector(root_fedcc)

# Create class labels (0–26 repeated for each client)
class_labels = np.tile(np.arange(num_classes), 3)  # shape (81,)
client_labels = np.repeat(['A', 'B', 'C'], num_classes)  # shape (81,)

all_points = torch.cat([
        combined_fedavg,  # A, B, C (source points)
        new_centroids_fedavg[num_classes:num_classes + 10],
        combined2_fedavg[num_classes:num_classes + 10],
        new_centroids2_fedavg,
        combined_fedcc,  # A, B, C (source points)
        new_centroids_fedcc[num_classes:num_classes + 10],
        combined2_fedcc[num_classes:num_classes + 10],
        new_centroids2_fedcc
    ], dim=0).numpy()


pca = PCA(n_components=2)
pca.fit(all_points)

embedding_result_fedavg = pca.transform(combined_fedavg)
pca_agg_result_fedavg = pca.transform(new_centroids_fedavg[num_classes:num_classes + 10])
embedding_result2_fedavg = pca.transform(combined2_fedavg)
pca_agg_result2_fedavg = pca.transform(new_centroids2_fedavg[num_classes:num_classes + 10])

embedding_result_fedcc = pca.transform(combined_fedcc)
pca_agg_result_fedcc = pca.transform(new_centroids_fedcc[num_classes:num_classes + 10])
embedding_result2_fedcc = pca.transform(combined2_fedcc)
pca_agg_result2_fedcc = pca.transform(new_centroids2_fedcc[num_classes:num_classes + 10])


pca_agg_result_fedcc = hungarian(embedding_result_fedcc, class_labels,
              num_classes, pca_agg_result_fedcc)

pca_agg_result2_fedcc = hungarian(embedding_result2_fedcc, class_labels,
              num_classes, pca_agg_result2_fedcc)


client_labels_tensor2 = np.array(['A'] * num_classes)  # Assuming 'A' represents the single client

# Colors: 27 distinct colors for classes using seaborn palette
class_palette = sns.color_palette("tab20", num_classes) + sns.color_palette("Set3", num_classes - 20)

# Markers for clients
client_markers = {'A': 'o', 'B': 's', 'C': '^'}

# Plot
# plt.figure(figsize=(14, 12), dpi=500)
fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=300)
fig.subplots_adjust(right=0.85, hspace=0.3, wspace=0.3)

# todo: CONTROL THE PLOT
flag = 'cc'

if flag == 'cc':
    embedding_result = embedding_result_fedcc
    pca_agg_result = pca_agg_result_fedcc
    embedding_result2 = embedding_result2_fedcc
    pca_agg_result2 = pca_agg_result2_fedcc
else:
    embedding_result = embedding_result_fedavg
    pca_agg_result = pca_agg_result_fedavg
    embedding_result2 = embedding_result2_fedavg
    pca_agg_result2 = pca_agg_result2_fedavg

# Plot 1: Original points
for i, (x, y) in enumerate(embedding_result):
    cls = class_labels[i]
    client = client_labels[i]
    axes[0][0].scatter(x, y, color=class_palette[cls], marker=client_markers[client], edgecolor='black', s=90)



axes[0][0].set_title("PCA of Client Centroids Before Agg1")
axes[0][0].set_xlabel("PC1")
axes[0][0].set_ylabel("PC2")

# Plot 2: FedAvg centroids
for i, (x, y) in enumerate(pca_agg_result):
    cls = class_labels[i]
    axes[0][1].scatter(x, y, color=class_palette[cls], marker='X', edgecolor='black', s=90)

axes[0][1].set_title("PCA of Client Centroids After Agg1")
axes[0][1].set_xlabel("PC1")
axes[0][1].set_ylabel("PC2")

# Plot 3: Original points before next agg
for i, (x, y) in enumerate(embedding_result2):
    cls = class_labels[i]
    client = client_labels[i]
    axes[1][0].scatter(x, y, color=class_palette[cls], marker=client_markers[client], edgecolor='black', s=90)

axes[1][0].set_title("PCA of Client Centroids Before Agg2")
axes[1][0].set_xlabel("PC1")
axes[1][0].set_ylabel("PC2")

# Plot 4: FedAvg centroids
for i, (x, y) in enumerate(pca_agg_result2):
    cls = class_labels[i]
    axes[1][1].scatter(x, y, color=class_palette[cls], marker='X', edgecolor='black', s=90)

axes[1][1].set_title("PCA of Client Centroids After Agg2")
axes[1][1].set_xlabel("PC1")
axes[1][1].set_ylabel("PC2")

# === Bind all plots to same axis scale ===
# all_xy = np.vstack([
#     embedding_result,
#     pca_result_fedavg,
#     embedding_result2,
#     pca_result_fedavg2
# ])
#
# x_min, x_max = all_xy[:, 0].min(), all_xy[:, 0].max()
# y_min, y_max = all_xy[:, 1].min(), all_xy[:, 1].max()
#
# for row in axes:
#     for ax in row:
#         ax.set_xlim(x_min, x_max)
#         ax.set_ylim(y_min, y_max)

fig.suptitle("Comparison of FedAvg Centroid Alignment on Client Features", fontsize=18, y=1.02)
plt.savefig("visual_comparison_full.png", dpi=300, bbox_inches="tight")
plt.show()

