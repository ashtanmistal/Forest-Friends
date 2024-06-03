"""
Author: Ashtan Mistal (derived from Charles R. Qi's implementation of PointNet++ in PyTorch)
Date: Jan 2024
"""
import json
import os
import pickle

import numpy as np
import pylas
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


def read_las_file(file_path):
    """
    Reads a .las file and extracts the necessary point cloud data.

    :param file_path: Path to the .las file to be read.
    :return: A tuple of numpy arrays (points, colors, labels).
    """

    # Open the .las file
    las_data = pylas.read(file_path)

    # Extract xyz coordinates
    points = np.vstack((las_data.x, las_data.y, las_data.z)).transpose()

    # Extract classification labels
    labels = las_data.classification

    # Check if color information is present and extract it
    if hasattr(las_data, 'red') and hasattr(las_data, 'green') and hasattr(las_data, 'blue'):
        colors = np.vstack((las_data.red, las_data.green, las_data.blue)).transpose()
    else:
        colors = np.zeros_like(points)  # If no color info, create a dummy array with zeros

    # Normalize or preprocess the points if needed
    # This would be based on the preprocessing done in the original S3DISDataLoader

    # Return the extracted data
    return points, colors, labels


def write_ply(filename, data, create=True):
    if create:
        # if the directory doesn't exist, create it
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("element vertex {}\n".format(len(data)))
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for row in data:
            f.write("{} {} {} {} {} {}\n".format(row[0], row[1], row[2], int(row[3]), int(row[4]), int(row[5])))


def filter_points_in_tree_radius(points, tree_radius, tree_center):
    """
    Filters out points that are not within the tree_radius of the tree_center.

    :param points:The points to filter. May contain additional columns after xyz.
    :param tree_radius: The radius of the tree to filter points by.
    :param tree_center: The center of the tree to filter points by.
    :return: The filtered points.
    """

    # Calculate the distance between each point and the center of the tree
    dist = np.linalg.norm(points[:, :3] - tree_center, axis=1)
    # Filter out points that are not within the tree_radius of the tree_center
    return points[dist <= tree_radius]


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


class UBCTreeDataset(Dataset):
    """
    Dataloader for individual trees from an .las file. This code is based on my own implementation of
    AerialLiDARDataLoader.py, in the repository https://github.com/ashtanmistal/LiDAR-DenseSeg, as well as the
    implementation of ModelNetDataLoader.py in the PointNet++ repository.
    """

    NUM_CLASSES = 2  # deciduous or coniferous
    CLASS_OF_INTEREST = "coniferous"  # this is arbitrary; makes conif. 1 and decid. 0
    OFFSET_X = 480000
    OFFSET_Y = 5455000
    ROTATION_DEGREES = 28.000  # This is the rotation of UBC's roads relative to true north.
    ROTATION_RADIANS = np.radians(ROTATION_DEGREES)
    INVERSE_ROTATION_MATRIX = np.array([[np.cos(ROTATION_RADIANS), np.sin(ROTATION_RADIANS), 0],
                                        [-np.sin(ROTATION_RADIANS), np.cos(ROTATION_RADIANS), 0],
                                        [0, 0, 1]])

    def __init__(self, root, args, split='train', process_data=True):
        self.root = root
        self.npoints = args.num_point
        self.process_data = process_data
        self.uniform = args.use_uniform_sample
        self.use_normals = args.use_normals
        # number of classes is 2: deciduous or coniferous
        self.num_category = 2

        # TODO process data to get self.list_of_points and self.list_of_labels
        # will need to run the machine learning model first to get the labels associated with each point

        if self.process_data:
            print('Processing data...')
            # TODO process data

            with open(self.save_path, 'wb') as f:
                pickle.dump([self.list_of_points, self.list_of_labels], f)
        else:
            print('Load processed data from %s...' % self.save_path)
            with open(self.save_path, 'rb') as f:
                self.list_of_points, self.list_of_labels = pickle.load(f)

    def __len__(self):
        return len(self.datapath)

    def _get_item(self, index):
        pass  # TODO implement this function

    def __getitem__(self, index):
        return self._get_item(index)


if __name__ == '__main__':
    import torch

    data = UBCTreeDataset(root="/data/ubc_trees", split="train", process_data=True)
    dataLoader = torch.utils.data.DataLoader(data, batch_size=8, shuffle=True)  # TODO play around with these values
    for point, label in dataLoader:
        print(point.shape)
        print(label.shape)
