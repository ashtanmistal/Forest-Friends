# Forest-Friends
Detection, positioning, and classification of biological tree families from airborne LiDAR data using PointNet++ and mean shift clustering. 

**_This project is a work-in-progress_.**

___

## Introduction

This repository is a supplement and extension of the [MinecraftUBC project](https://github.com/ashtanmistal/minecraftUBC).

LiDAR data (by itself) is great for visualization of forests, but is lacking in analysis capabilities in a tree-by-tree basis without additional processing. This project aims to gather information about a given area by using the LiDAR data to cluster point clouds into individual trees, calculating their height and diameter, and then classifying each tree into a family based on the clustered point cloud.

Within the context of the MinecraftUBC project, this information can be used to generate a more realistic forest environment by choosing voxel type based on tree family, therefore providing more variance and realism in a forest. This additionally provides a method to place tree trunks for each tree; information that is not present in the LiDAR data given the lack of visibility of the forest floor.

## Technical Details

The clustering is performed using a self-written implementation of [*A Robust Stepwise Clustering Approach to Detect Individual Trees in Temperate Hardwood Plantations using Airborne LiDAR Data*](https://doi.org/10.3390/rs15051241) by Shao et al. (2023).

For training of a [PointNet++ classification model](https://doi.org/10.48550/arXiv.1706.02413), each tree in the training dataset is supplemented by a [database provided by UBCGeodata](https://github.com/UBCGeodata/ubc-geospatial-opendata) that provides information about the tree's taxonomy, common name, and type. This database is not complete for each tree that is logged, and lacks complete data about tree types and families, and so a pre-processing script is used to fill in missing data for each logged tree from the taxonomy using the [GBIF (Global Biodiversity Information Facility) API](https://www.gbif.org/developer/summary), with GPT-3 used as a last resort for missing data in deciduous / coniferous classification if the GBIF API does not return a result.

The tree type is then used as the label during classification training in PointNet++. The geospatial database only covers a small subset of the trees in the LiDAR dataset -- the regional park surrounding UBC does not have associated data -- and as a result the goal is to determine the tree type for *every* tree in the LiDAR dataset using the trained model, thus providing a robust model for tree detection and classification from raw LiDAR data.
