import pyproj
import numpy as np
import math

ROTATION_DEGREES = 28.000  # This is the rotation of UBC's roads relative to true north.
ROTATION_RADIANS = math.radians(ROTATION_DEGREES)
INVERSE_ROTATION_MATRIX = np.array([[math.cos(ROTATION_RADIANS), math.sin(ROTATION_RADIANS), 0],
                                    [-math.sin(ROTATION_RADIANS), math.cos(ROTATION_RADIANS), 0],
                                    [0, 0, 1]])
BLOCK_OFFSET_X = 480000
BLOCK_OFFSET_Z = 5455000

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