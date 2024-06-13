"""
Author: Ashtan Mistal (derived from Charles R. Qi's implementation of PointNet++ in PyTorch)
Date: Jan 2024
"""
import json
import os
import pickle

import numpy as np
import pandas as pd
import pylas
import torch
from scipy.spatial import cKDTree
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from tqdm import tqdm


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


class TreeLiDARDataLoader(Dataset):
    """
    Dataloader for individual trees from an .las file. This code is based on my own implementation of
    AerialLiDARDataLoader.py, in the repository https://github.com/ashtanmistal/LiDAR-DenseSeg, as well as the
    implementation of ModelNetDataLoader.py in the PointNet++ repository.
    """

    NUM_CLASSES = 2  # deciduous or coniferous
    CLASS_OF_INTEREST = "coniferous"  # this is arbitrary; makes conif. 1 and decid. 0
    # TODO can classes be strings or do we need to convert to integers?

    CLASS_ENUM = {"deciduous": 0, "coniferous": 1}

    OFFSET_X = 480000
    OFFSET_Y = 5455000
    ROTATION_DEGREES = 28.000  # This is the rotation of UBC's roads relative to true north.
    ROTATION_RADIANS = np.radians(ROTATION_DEGREES)
    INVERSE_ROTATION_MATRIX = np.array([[np.cos(ROTATION_RADIANS), np.sin(ROTATION_RADIANS), 0],
                                        [-np.sin(ROTATION_RADIANS), np.cos(ROTATION_RADIANS), 0],
                                        [0, 0, 1]])

    def __init__(self, root, args, split='train', process_data=True):
        self.root = root
        # self.npoints = args.num_point
        self.process_data = process_data
        self.uniform = args.use_uniform_sample
        self.use_normals = args.use_normals
        self.npoints = args.num_point
        # number of classes is 2: deciduous or coniferous
        self.num_category = 2

        self.label_path = (r"C:\Users\Ashtan Mistal\OneDrive - "
                           r"UBC\School\2023S\minecraftUBC\resources\ubcv_campus_trees_processed.csv")

        assert (split == 'train' or split == 'test')

        if self.process_data:
            print('Processing data %s (only running in the first time)...' % self.root)
            """
            In order to process the data, we need to read through the data directory and get the points with their 
            associated labels. Each dataset was clustered separately, so we'll have to bring together all the points
            into self.points. Self.labels should be automatically incremented. 
            
            For the training data, we need to read the preprocessing .csv file and find the label with the closest
            cluster center. All points associated with that label should be returned when that index is accessed in the
            getter. 
            """

            # for each file in the save path that ends in .csv we will have an associated _clustered_points.csv,
            # _cluster_labels.csv, _cluster_centers.csv.

            # self.num_labels is the maximum label found after we have read all the .csv files.
            # An alternative is to save the maximum label to a .txt file as we do determine this in clustering.py

            # we do not need to put all points into a main "points" array first; we can just go through each dataset
            # and add each label and their corresponding points to the arrays. This should require only one pass through
            # the dataset.
            # The points are sorted by label, so we can use np.searchsorted instead of np.where to find the indices.

            # self.list_of_points = [None] * self.num_labels
            # self.list_of_labels = [None] * self.num_labels

            self.list_of_points = []
            self.list_of_labels = []

            tree_labels = []
            tree_points = []

            label_data = pd.read_csv(self.label_path, header=0)
            for row in label_data.itertuples(index=False):
                xz = np.array([row[5], row[6]])
                tree_labels.append(self.CLASS_ENUM[row[2]])
                tree_points.append(xz)

            # make a kdtree of the tree points
            tree_points = np.array(tree_points)
            tree_kdtree = cKDTree(tree_points)

            for filename in os.listdir(self.root):
                if filename.endswith('_label_data.csv'):
                    # label_data = pd.read_csv(os.path.join(self.root, filename), header=None)
                    # clustered_points_file = filename.replace('_label_data.csv', '_clustered_points.csv')
                    # clustered_points = pd.read_csv(os.path.join(self.root, clustered_points_file), header=None)
                    # cluster_labels_file = filename.replace('_label_data.csv', '_cluster_labels.csv')
                    # cluster_labels = pd.read_csv(os.path.join(self.root, cluster_labels_file), header=None)
                    # cluster_labels = cluster_labels.to_numpy().astype(np.int32).flatten()

                    # use numpy instead of pandas
                    label_data = np.genfromtxt(os.path.join(self.root, filename), delimiter=',')
                    clustered_points_file = filename.replace('_label_data.csv', '_clustered_points.csv')
                    clustered_points = np.genfromtxt(os.path.join(self.root, clustered_points_file), delimiter=',')
                    cluster_labels_file = filename.replace('_label_data.csv', '_cluster_labels.csv')
                    cluster_labels = np.genfromtxt(os.path.join(self.root, cluster_labels_file), delimiter=',').astype(
                        np.int32)

                    for row in label_data:
                        # the label data here is [number (label index), center_x, center_z, height]. no header.
                        # so query the KDtree for the closest point to the center_x, center_z within 3m
                        # add the points to the list of points and the label (from tree_labels) to the list of labels
                        center_x = row[1]
                        center_z = row[2]

                        _, idx = tree_kdtree.query([center_x, center_z], distance_upper_bound=3, k=1)
                        if idx == tree_points.shape[0]:
                            continue  # this is a tree that we must use the trained model to predict
                        named_label = tree_labels[idx]
                        points = clustered_points[np.where(cluster_labels == row[0])]

                        if self.uniform:
                            points = farthest_point_sample(points, self.npoints)
                        else:
                            points = points[0:self.npoints, :]

                        self.list_of_points.append(points)
                        self.list_of_labels.append(named_label)

            self.num_labels = len(self.list_of_labels)

            with open(os.path.join(self.root, 'processed_data.pkl'), 'wb') as f:
                pickle.dump([self.list_of_points, self.list_of_labels], f)
        else:
            print('Load processed data from %s...' % self.root)
            with open(os.path.join(self.root, 'processed_data.pkl'), 'rb') as f:
                self.list_of_points, self.list_of_labels = pickle.load(f)

        # Split data into train and test sets
        train_points, test_points, train_labels, test_labels = train_test_split(
            self.list_of_points, self.list_of_labels, test_size=0.2, random_state=42)

        if split == 'train':
            self.list_of_points = train_points
            self.list_of_labels = train_labels
        else:
            self.list_of_points = test_points
            self.list_of_labels = test_labels

        self.num_labels = len(self.list_of_labels)




    def __len__(self):
        return self.num_labels

    def _get_item(self, index):
        """
        Gathers the data at label index and returns the corresponding points and label.
        """
        if self.process_data:
            point_set, label = self.list_of_points[index], self.list_of_labels[index]
            # TODO do we need an "else"? Given that we are loading self.list_of_points and self.list_of_labels
            # in the already processed data too.
        else:
            point_set = self.list_of_points[index]
            label = self.list_of_labels[index]

        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        if not self.use_normals:
            point_set = point_set[:, 0:3]

        return point_set.astype(np.float32), label

    def __getitem__(self, index):
        return self._get_item(index)


if __name__ == '__main__':
    import torch

    data = TreeLiDARDataLoader(root="/data/ubc_trees", split="train", process_data=True)  # TODO update root
    # TODO add args
    dataLoader = torch.utils.data.DataLoader(data, batch_size=8, shuffle=True)  # TODO play around with these values
    for point, label in dataLoader:
        print(point.shape)
        print(label.shape)
