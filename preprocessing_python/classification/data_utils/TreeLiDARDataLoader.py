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
        # TODO get the save path and assign directly (removing it from args)

        self.save_path = args.save_path

        if self.process_data:
            print('Processing data %s (only running in the first time)...' % self.save_path)
            # TODO process data
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

            # self.list_of_points = [None] * self.num_labels
            # self.list_of_labels = [None] * self.num_labels

            with open(self.save_path, 'wb') as f:
                pickle.dump([self.list_of_points, self.list_of_labels], f)  # this is going to be a large file
                # TODO add the pickle to .gitignore once it is created
        else:
            print('Load processed data from %s...' % self.save_path)
            with open(self.save_path, 'rb') as f:
                self.list_of_points, self.list_of_labels = pickle.load(f)

    def __len__(self):
        return len(self.num_labels)  # TODO calculate num_labels (see above)

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

        return point_set, label[0]

    def __getitem__(self, index):
        return self._get_item(index)


if __name__ == '__main__':
    import torch

    data = UBCTreeDataset(root="/data/ubc_trees", split="train", process_data=True)  # TODO update root
    # TODO add args
    dataLoader = torch.utils.data.DataLoader(data, batch_size=8, shuffle=True)  # TODO play around with these values
    for point, label in dataLoader:
        print(point.shape)
        print(label.shape)
