import os

import numpy as np
import pylas
from sklearn.cluster import MeanShift
from tqdm import tqdm

import utils

data_dir = r"C:\Users\Ashtan Mistal\OneDrive - UBC\School\2023S\minecraftUBC\resources"
save = True


def gather_lidar_data():
    tree_points = []

    # lidar data is in data/las folder
    lidar_directory = os.path.join(data_dir, "las")

    for filename in tqdm(os.listdir(lidar_directory)):
        if filename.endswith(".las"):
            x, y, z, r, g, b = utils.preprocess_dataset(
                pylas.read(os.path.join(lidar_directory, filename)),
                5,  # high vegetation label
            )
            if len(x) == 0:
                continue
            tree_points.append(np.vstack((x, z, y, r, g, b)).transpose())

    tree_points = np.vstack(tree_points)
    return tree_points


def vertical_strata_analysis(cluster_centers, meanshift_labels, x, y, z):
    """
    This function performs the vertical strata analysis on the clusters to determine which clusters are crown clusters
    :param cluster_centers: mean shift cluster centers
    :param meanshift_labels: array of labels for each point
    :param x: Array of x coordinates in the chunk
    :param y: Array of y coordinates in the chunk (height)
    :param z: Array of z coordinates in the chunk
    :return: crown_clusters (list of cluster indices), non_ground_points (list of arrays of non-ground points),
    tree_cluster_centers (list of tree cluster centers), tree_clusters (list of tree cluster indices)
    """
    valid_indices = meanshift_labels != -1
    unique_labels, inverse_indices = np.unique(meanshift_labels[valid_indices], return_inverse=True)

    # Pre-allocate lists for clusters and centers
    # num_clusters = len(unique_labels)
    # non_ground_points = []
    crown_clusters = []
    crown_cluster_centers = []
    tree_clusters = []
    tree_cluster_centers = []

    ngx = []
    ngy = []
    ngz = []

    # Loop over the unique clusters
    for idx, label in enumerate(unique_labels):
        # Indices of the current cluster
        cluster_mask = inverse_indices == idx
        cluster_x = x[valid_indices][cluster_mask]
        cluster_y = y[valid_indices][cluster_mask]
        cluster_z = z[valid_indices][cluster_mask]

        vertical_gap_y = np.max(cluster_y) * 0.3
        non_ground_mask = cluster_y > vertical_gap_y

        # Append non-ground points for this cluster
        # non_ground_points.append(
        #     np.stack([cluster_x[non_ground_mask], cluster_y[non_ground_mask], cluster_z[non_ground_mask]], axis=-1))
        ngx.extend(cluster_x[non_ground_mask])
        ngy.extend(cluster_y[non_ground_mask])
        ngz.extend(cluster_z[non_ground_mask])

        # Analyze vertical length ratio (VLR) of the cluster
        vlr = (np.max(cluster_y) - np.min(cluster_y)) / np.max(cluster_y)
        cutoff = 0.5

        # Categorize cluster based on VLR
        if vlr < cutoff:
            crown_clusters.append(label)
            crown_cluster_centers.append(cluster_centers[label])
        else:
            tree_clusters.append(label)
            tree_cluster_centers.append(cluster_centers[label])

    # return crown_clusters, non_ground_points, tree_cluster_centers, tree_clusters
    return crown_clusters, np.array(ngx), np.array(ngy), np.array(ngz), tree_cluster_centers, tree_clusters


def cluster(x, y, z, r, g, b, perform_vertical_strata_analysis=False):
    stacked_xz = np.vstack((x, z)).transpose()
    bandwidth = 5  # manually set bandwidth based on the approximate diameter of a tree
    ms = MeanShift(bandwidth=bandwidth, cluster_all=False, n_jobs=-1, bin_seeding=True, min_bin_freq=1000)
    print("Fitting meanshift...")
    ms.fit(stacked_xz)
    ms_labels = ms.labels_
    cluster_centers = ms.cluster_centers_

    # remove points that are not in a cluster
    clustered = np.where(ms_labels != -1)
    print(f"Number of unclustered points: {len(np.where(ms_labels == -1)[0])}, out of {len(ms_labels)}")
    x, y, z = x[clustered], y[clustered], z[clustered]
    r, g, b = r[clustered], g[clustered], b[clustered]
    ms_labels = ms_labels[clustered]

    # plot the clusters
    # utils.plot_clusters(
    #     np.vstack((x, y, z)).transpose(),
    #     ms_labels, cluster_centers, "None")
    # import pdb; pdb.set_trace()

    if perform_vertical_strata_analysis:
        # TODO it will be beneficial to only look for the nearest tree cluster within 5m of the crown cluster
        print(f"Number of clusters: {len(np.unique(ms_labels))}. Performing vertical strata analysis...")
        crown_cl, ngx, ngy, ngz, cl_centers, tree_cl = vertical_strata_analysis(cluster_centers, ms_labels, x, y, z)
        print(f"Number of crown clusters: {len(crown_cl)}; Number of tree clusters: {len(tree_cl)}")
        # Now we need to assign the crown clusters to the nearest tree cluster
        if len(tree_cl) > 0:
            for cluster in crown_cl:
                cluster_indices = np.where(ms_labels == cluster)
                cluster_x, cluster_y, cluster_z = ngx[cluster], ngy[cluster], ngz[cluster]
                # for point in np.array([cluster_x, cluster_y, cluster_z]).T:
                concatenated_points = np.array([cluster_x, cluster_z]).T
                for point in concatenated_points:
                    # find the nearest tree cluster
                    # comparison_point = np.array([point[0], point[2]])
                    nearest_cluster = tree_cl[
                        np.argmin(np.linalg.norm(np.array(cl_centers) - point, axis=1))]
                    # assign the point to that cluster
                    ms_labels[cluster_indices] = nearest_cluster
            # re-calculating the cluster centers and calculating the height of each tree
            unique_clusters = np.unique(ms_labels)
            num_clusters = unique_clusters.size

            cluster_centers = np.empty((num_clusters, 2))  # Two columns for x and z averages
            cluster_heights = np.empty(num_clusters)

            # Fill the arrays
            for i, cluster in enumerate(unique_clusters):
                cluster_indices = np.where(ms_labels == cluster)
                cluster_x, cluster_y, cluster_z = x[cluster_indices], y[cluster_indices], z[cluster_indices]
                cluster_centers[i] = [np.average(cluster_x), np.average(cluster_z)]
                cluster_heights[i] = np.max(cluster_y)
        else:
            raise ValueError("No tree clusters found. This is likely due to the chunk being empty.")
        ds = np.vstack((x, y, z, r, g, b)).transpose()
        return ds, ms_labels, cluster_centers, cluster_heights
    else:
        ds = np.vstack((x, y, z, r, g, b)).transpose()

        # calculate the height of each cluster
        cluster_heights = np.empty(cluster_centers.shape[0])
        for i, cluster in enumerate(cluster_centers):
            cluster_indices = np.where(ms_labels == i)
            cluster_heights[i] = np.max(y[cluster_indices])
        return ds, ms_labels, cluster_centers, cluster_heights


def main():
    max_label = 0

    lidar_directory = os.path.join(data_dir, "las")

    completed_datasets = [
    ]

    for filename in os.listdir(lidar_directory):
        if filename.endswith(".las") and filename not in completed_datasets:
            print("Processing", filename)
            x, y, z, r, g, b = utils.preprocess_dataset(
                pylas.read(os.path.join(lidar_directory, filename)),
                5,  # high vegetation label
            )
            if len(x) == 0:
                continue

            # ignore half of the points to speed up processing during initial model tuning
            x = x[::2]
            y = y[::2]
            z = z[::2]
            r = r[::2]
            g = g[::2]
            b = b[::2]

            try:
                clustered_points, labels, cluster_centers, cluster_heights = cluster(x, y, z, r, g, b)
                # utils.plot_clusters(clustered_points, labels, cluster_centers, filename)
                labels += max_label
                max_label = np.max(labels) + 1
                if save:
                    base = os.path.splitext(filename)[0]
                    if not os.path.exists(os.path.join(data_dir, "clustered_points")):
                        os.makedirs(os.path.join(data_dir, "clustered_points"))
                    np.savetxt(os.path.join(data_dir, "clustered_points", base + "_clustered_points.csv"),
                               clustered_points, delimiter=",")
                    np.savetxt(os.path.join(data_dir, "clustered_points", base + "_cluster_labels.csv"), labels,
                               delimiter=",")
                    np.savetxt(os.path.join(data_dir, "clustered_points", base + "_cluster_centers.csv"),
                               cluster_centers, delimiter=",")
                    np.savetxt(os.path.join(data_dir, "clustered_points", base + "_cluster_heights.csv"),
                               cluster_heights, delimiter=",")
            except ValueError as e:
                print(e)
                print("Skipping", filename)
    print("Clustering complete.")


if __name__ == "__main__":
    main()
