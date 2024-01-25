This file contains the planning and design documentation for the preprocessing and clustering steps of Forest Friends.

# Preprocessing

The following steps are taken to preprocess the raw LiDAR data:
- For each dataset partition:
  - Remove all non-tree points (based on classification)
  - Only keep x, y, z, r, g, b columns
  - Rotate and translate points to align with coordinates of Minecraft world
    - NOTE this is a specific step that must be modified if the LiDAR data is from a different source. I will make this robust and allow for an (X, Y, Z -> X', Y', Z') transformation function to be passed in as a parameter.
  - Remove height from trees: For the clustering, all the trees need to be relative to the ground. As a result, we'll need the triangulated DEM of the area to subtract the height of each tree from.
    - For this *particular* project, the DEM is calculated separately in the MinecraftUBC repository given it is also used for additional tasks there too (voxelization and colorization of the DEM). **It is therefore assumed, for this project, that the DEM is already calculated and available.**
  - Merge all partitions into one dataset for downstream processing

Preprocessing the .csv file:
- This is where we need to fill in missing columns for each tree
- If both the common name and the taxa are missing, we can't do anything about it (remove tree from training set)
- For each unique common name / taxa pair:
  - If one is missing, use the GBIF API to fill in the missing data for all trees with that common name / taxa pair
  - Determine the tree type from the common name / taxa pair (use the GBIF API to get the tree type from the taxa)
  - Given we have (taxa or common name), and *either one* can be missing but there's a 1:1 mapping between them, we can use the GBIF API to fill in the missing data for all trees with that common name or taxa.
  - We'll also need to take a look at the xyz data, convert coordinate systems if necessary, and apply the same rotation and translation as we did for the LiDAR data.
  - Save the .csv file as a new file with the filled in data, keeping only the columns necessary for training. 

# Clustering

The clustering really is just another "preprocessing" step for the PointNet++ down the line. This is straightforward (as I've already implemented it); some tasks may have to be done in parallel to speed up the process. The previous implementation did not take full CUDA advantage, so code modifications may be necessary to speed up the process.

___

Once the clustering and preprocessing is done, there is further preprocessing that is necessary. 
Now that we have the LiDAR data and the tree data, we need to match them up.
It'll probably be good to make a KDTree of the tree location and tree type from the CSV file. 
Then, for each point in the LiDAR data, we can find the nearest tree and assign it the tree type of that tree (the "label" for the point).

The next step is to be able to access the data in a way that is easy to use for the PointNet++.
We need to be able to be given an index and return the point cloud for that tree along with the label.
This specific step will be done in the DataLoader class for the PointNet++ model.