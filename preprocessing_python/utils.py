import pyproj
import numpy as np
import math
import matplotlib.pyplot as plt

ROTATION_DEGREES = 28.000  # This is the rotation of UBC's roads relative to true north.
ROTATION_RADIANS = math.radians(ROTATION_DEGREES)
INVERSE_ROTATION_MATRIX = np.array([[math.cos(ROTATION_RADIANS), math.sin(ROTATION_RADIANS), 0],
                                    [-math.sin(ROTATION_RADIANS), math.cos(ROTATION_RADIANS), 0],
                                    [0, 0, 1]])
BLOCK_OFFSET_X = 480000
BLOCK_OFFSET_Z = 5455000
HEIGHT_OFFSET = 59


def convert_lat_long_to_x_z(lat, long, return_int=True):
    """
    Converts the given latitude and longitude coordinates to Minecraft x and z coordinates. Uses a pipeline to convert
    from EPSG:4326 (lat/lon) to EPSG:26910 (UTM zone 10N).
    :param lat: the latitude coordinate
    :param long: the longitude coordinate
    :param return_int: whether to return the coordinates as integers or not
    :return: the Minecraft x and z coordinates of the given latitude and longitude
    """
    pipeline = "+proj=pipeline +step +proj=axisswap +order=2,1 +step +proj=unitconvert +xy_in=deg +xy_out=rad +step " \
               "+proj=utm +zone=10 +ellps=GRS80"
    transformer = pyproj.Transformer.from_pipeline(pipeline)
    transformed_x, transformed_z = transformer.transform(lat, long)
    x, z, _ = np.matmul(INVERSE_ROTATION_MATRIX, np.array([transformed_x - BLOCK_OFFSET_X,
                                                           transformed_z - BLOCK_OFFSET_Z,
                                                           1]))
    z = -z  # flip z axis to match Minecraft

    if return_int:
        return int(x), int(z)
    else:
        return x, z


def preprocess_dataset(lidar_ds, label_to_keep):
    """
    Preprocesses the given dataset by removing all points that are not of the given label, and then rotating the
    dataset to match Minecraft's orientation.
    :param lidar_ds: the dataset to preprocess
    :param label_to_keep: the label (e.g. 2 for ground terrain) to keep
    :return: the maximum and minimum x and z coordinates of the dataset, and the x, y, and z coordinates of the dataset
    """
    initial_x, initial_z, initial_y, labels = lidar_ds.x, lidar_ds.y, lidar_ds.z, lidar_ds.classification
    initial_red, initial_green, initial_blue = lidar_ds.red, lidar_ds.green, lidar_ds.blue
    indices_to_keep = labels == label_to_keep
    # Filter the data by keeping only the indices of the specified label
    filtered_x = initial_x[indices_to_keep]
    filtered_y = initial_y[indices_to_keep]
    filtered_z = initial_z[indices_to_keep]
    filtered_red = initial_red[indices_to_keep]
    filtered_green = initial_green[indices_to_keep]
    filtered_blue = initial_blue[indices_to_keep]
    dataset = np.matmul(INVERSE_ROTATION_MATRIX, np.array([filtered_x - BLOCK_OFFSET_X,
                                                           filtered_z - BLOCK_OFFSET_Z,
                                                           filtered_y - HEIGHT_OFFSET]))
    if dataset.shape[1] == 0:
        return [], [], [], [], [], []
    rotated_x, rotated_z, rotated_y = dataset[0], -dataset[1], dataset[2]
    return rotated_x, rotated_y, rotated_z, filtered_red, filtered_green, filtered_blue


def plot_clusters(clustered_points, labels, cluster_centers):
    """
    Plots the clustered points, ignoring the height values y (plotting x and z coordinates only).
    This is intended to be called after Mean Shift Clustering in clustering.py
    Uses hexbin to plot the clusters, coloring by label, and plotting cluster centers on top of the hexbin.
    :param clustered_points: the points to plot
    :param labels: Labels corresponding to the datapoints
    :param cluster_centers: the centers of the clusters
    """
    x = clustered_points[:, 0]
    z = clustered_points[:, 2]

    plt.figure(figsize=(10, 8))

    plt.hexbin(x, z, C=labels, gridsize=50, cmap='inferno', reduce_C_function=np.mean)
    plt.colorbar(label='Cluster Label')

    for center in cluster_centers:
        plt.plot(center[0], center[2], 'bo', markersize=10, label='Cluster Center')

    plt.xlabel('X Coordinate')
    plt.ylabel('Z Coordinate')
    plt.title('Clustered Points with Cluster Centers')
    plt.legend()
    plt.show()



