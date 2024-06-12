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

As discussed in the primary README, the clustering is performed using a self-written implementation of [*A Robust Stepwise Clustering Approach to Detect Individual Trees in Temperate Hardwood Plantations using Airborne LiDAR Data*](https://doi.org/10.3390/rs15051241) by Shao et al. (2023).

However, there are some modifications that need to be made to the algorithm to better suit to our dataset. Namely, fine-tuning of the vertical length ratio (VLR) of the cluster and the bandwidth of the mean shift clustering.

After visualizing results, standalone mean shift clustering produces better results than with the vertical strata analysis. The vertical strata analysis pairs far too many tree clusters together such that there are some tree clusters that contain the vast majority of the points. The same does not occur with the standalone mean shift clustering. With some fine tuning (and thus manual setting) of the bandwidth, the standalone mean shift clustering produces better results.

![img.png](img.png)

Above is an image with the vertical strata analysis integrated into the clustering algorithm. The points on the left have a more negative height value (as seen in the rightmost inset) and as a result get clustered together in the vertical strata analysis. This is consistent across similar datasets. The points on the right have a more positive height value - both in absolute as well as the fact that the trees are taller - and therefore remain their own clusters. It is difficult to determine a good VLR that will work for all tree types seen in the UBC dataset. As a result, the standalone mean shift clustering is used for the final results.

![img_1.png](img_1.png)

Above is an example of the same dataset but with the standalone mean shift clustering. The clusters are more evenly distributed and the trees are more accurately represented. Note that in both datasets, `cluster_all` was set to `False`, and as a result not all datapoints are assigned to a tree. These points are still plotted in the overall Minecraft world but are not used for the purposes of tree trunk positioning and tree type classification.