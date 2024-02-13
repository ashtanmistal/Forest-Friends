import pylas
from sklearn.cluster import MeanShift, DBSCAN
import pandas as pd
import numpy as np
import os

from tqdm import tqdm

from preprocessing import data_dir
from src import utils

MIN_BIN_FREQ = 60


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
    non_ground_points = []
    crown_clusters = []
    crown_cluster_centers = []
    tree_clusters = []
    tree_cluster_centers = []
    for cluster in np.unique(meanshift_labels):
        # if the label is -1, continue
        if cluster == -1:
            continue
        cluster_indices = np.where(meanshift_labels == cluster)
        cluster_x, cluster_y, cluster_z = x[cluster_indices], y[cluster_indices], z[cluster_indices]
        vertical_gap_y = np.max(cluster_y) * 0.3  # this 0.3 is based on the study
        non_ground_indices = np.where(cluster_y > vertical_gap_y)
        # we can ignore the ground residuals now. We just want to keep the crown and stem points
        non_ground_points.append(
            np.array([cluster_x[non_ground_indices], cluster_y[non_ground_indices], cluster_z[non_ground_indices]]))

        # Now we need to determine if this is a crown cluster or a tree cluster. To do this we analyze the vertical
        # length ratio of the cluster.
        vlr = (np.max(cluster_y) - np.min(cluster_y)) / np.max(cluster_y)
        # A high VLR cluster is a tree cluster, a low VLR cluster is a crown cluster
        cutoff = 0.62  # Modified cutoff value from the study to try and get more tree clusters given the fact that we
        # a. are taking into account less data points, and b. are limiting the meanshift to a per-chunk basis
        if vlr < cutoff:
            crown_clusters.append(cluster)  # crown cluster. We need to assign these points to the nearest tree
            # cluster later.
            crown_cluster_centers.append(cluster_centers[cluster])
        else:
            tree_clusters.append(cluster)
            tree_cluster_centers.append(cluster_centers[cluster])

    return crown_clusters, non_ground_points, tree_cluster_centers, tree_clusters


def cluster(ds):
    x = ds[:, 0]
    y = ds[:, 1]
    z = ds[:, 2]
    r = ds[:, 3]
    g = ds[:, 4]
    b = ds[:, 5]

    ms = MeanShift(bin_seeding=True, min_bin_freq=MIN_BIN_FREQ, cluster_all=False, max_iter=63, n_jobs=15)
    print("Fitting meanshift...")
    ms.fit(np.vstack((x, z)).transpose())  # we don't care about the height or color for this part
    ms_labels = ms.labels_
    cluster_centers = ms.cluster_centers_
    # remove points that are not in a cluster
    non_ground_indices = np.where(ms_labels != -1)
    x, y, z = x[non_ground_indices], y[non_ground_indices], z[non_ground_indices]
    r, g, b = r[non_ground_indices], g[non_ground_indices], b[non_ground_indices]
    ms_labels = ms_labels[non_ground_indices]
    print("Performing vertical strata analysis...")
    crown_clusters, non_ground_points, tree_cluster_centers, tree_clusters = vertical_strata_analysis(cluster_centers,
                                                                                                      ms_labels, x, y,
                                                                                                      z)
    # Now we need to assign the crown clusters to the nearest tree cluster
    if len(tree_clusters) > 0:
        for cluster in crown_clusters:
            cluster_indices = np.where(ms_labels == cluster)
            cluster_x, cluster_y, cluster_z = non_ground_points[cluster]
            for point in np.array([cluster_x, cluster_y, cluster_z]).T:
                # find the nearest tree cluster
                comparison_point = np.array([point[0], point[2]])
                nearest_cluster = tree_clusters[
                    np.argmin(np.linalg.norm(np.array(tree_cluster_centers) - comparison_point, axis=1))]
                # assign the point to that cluster
                ms_labels[cluster_indices] = nearest_cluster
        # re-calculating the cluster centers and calculating the height of each tree
        cluster_centers = []
        cluster_heights = []
        for cluster in np.unique(ms_labels):
            cluster_indices = np.where(ms_labels == cluster)
            cluster_x, cluster_y, cluster_z = x[cluster_indices], y[cluster_indices], z[cluster_indices]
            cluster_centers.append(np.array([np.average(cluster_x), np.average(cluster_z)]))
            cluster_heights.append(np.max(cluster_y))
        cluster_centers = np.array(cluster_centers)
        cluster_heights = np.array(cluster_heights)
    else:
        raise ValueError("No tree clusters found. This is likely due to the chunk being empty.")
    ds = np.vstack((x, y, z, r, g, b)).transpose()
    return ds, ms_labels, cluster_centers, cluster_heights


def main():

    gathered_clustered_points = []
    gathered_labels = []
    gathered_cluster_centers = []
    gathered_cluster_heights = []
    max_label = 0

    lidar_directory = os.path.join(data_dir, "las")

    for filename in tqdm(os.listdir(lidar_directory)):
        if filename.endswith(".las"):
            x, y, z, r, g, b = utils.preprocess_dataset(
                pylas.read(os.path.join(lidar_directory, filename)),
                5,  # high vegetation label
            )
            if len(x) == 0:
                continue

            tree_points = np.vstack((x, z, y, r, g, b)).transpose()

            clustered_points, labels, cluster_centers, cluster_heights = cluster(tree_points)
            # There will be conflicts in the label numbers between quadrants, so we need to add an offset to the labels
            labels += max_label
            max_label = np.max(labels) + 1
            gathered_clustered_points.append(clustered_points)
            gathered_labels.append(labels)
            gathered_cluster_centers.append(cluster_centers)
            gathered_cluster_heights.append(cluster_heights)
    # combine the clustered points from all quadrants
    clustered_points = np.vstack(gathered_clustered_points)
    labels = np.concatenate(gathered_labels)
    cluster_centers = np.vstack(gathered_cluster_centers)
    cluster_heights = np.concatenate(gathered_cluster_heights)

    # save the clustered points to a csv file
    np.savetxt(os.path.join(data_dir, "clustered_points.csv"), clustered_points, delimiter=",")
    np.savetxt(os.path.join(data_dir, "cluster_labels.csv"), labels, delimiter=",")
    np.savetxt(os.path.join(data_dir, "cluster_centers.csv"), cluster_centers, delimiter=",")
    np.savetxt(os.path.join(data_dir, "cluster_heights.csv"), cluster_heights, delimiter=",")
    print("Clustering complete.")


if __name__ == "__main__":
    main()
