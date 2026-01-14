# -*- coding: utf-8 -*-
"""
Code description.
"""
# Author: Zhaoliang Zheng <zhz03@g.ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

import math

# Prepare the input file.
# XODR_FILE = "/home/zzl/zhaoliang/zhz03_github/Multi-Mod_Sensor_Config_Lib/sensor_configurator/utils/Town10HD_Opt.xodr"
XODR_FILE = "/home/handsomeyun/Downloads/Town10HD_Opt.xodr"
# XODR_FILE = "./s_metrics/maps/example_data/xodr/Town06.xodr"
# XODR_FILE = "./s_metrics/maps/example_data/xodr/Town07.xodr"
# XODR_FILE = "./s_metrics/maps/example_data/xodr/Town05.xodr"
# XODR_FILE = "./s_metrics/maps/example_data/xodr/Town04.xodr"
# XODR_FILE = "./s_metrics/maps/example_data/xodr/Town03.xodr"

def to_color(r, g, b):
    return '#{:02x}{:02x}{:02x}'.format(r, g, b)

# Prepare the colors.
DRIVING_COLOR = (135, 151, 154)
TYPE_COLOR_DICT = {
    "shoulder": (136, 158, 131),
    "border": (84, 103, 80),
    "driving": DRIVING_COLOR,
    "stop": (128, 68, 59),
    "none": (236, 236, 236),
    "restricted": (165, 134, 88),
    "parking": (0, 255, 0),
    "crosswalk": (255, 0, 0),
    "median": (119, 155, 88),
    "biking": (108, 145, 125),
    "sidewalk": (106, 159, 170),
    "curb": (30, 49, 53),
    "exit": DRIVING_COLOR,
    "entry": DRIVING_COLOR,
    "onramp": DRIVING_COLOR,
    "offRamp": DRIVING_COLOR,
    "connectingRamp": DRIVING_COLOR,
    "onRamp": DRIVING_COLOR,
    "bidirectional": DRIVING_COLOR,
}
TYPE_COLOR_DICT = {k: to_color(*v) for k, v in TYPE_COLOR_DICT.items()}
COLOR_CENTER_LANE = "#FFC500"
COLOR_REFERECE_LINE = "#0000EE"

TYPE_COLOR_DICT_VOXEL = {
    "driving": "orange",
    "sidewalk": "green",
    "median": "gray",
    "shoulder": "blue"
}

def reverse_y_direction(position):
    # convert the CARLA "left-handed" coordinate system to the "right-handed" coordinate system
    return (position[0], -position[1], position[2])

def convert_pyr2rpy_in_radians(rotation):
    # pitch,yaw,roll in degree to  roll, pitch, yaw in radians
    # also convert the CARLA "left-handed" coordinate system to the "right-handed" coordinate system 
    pitch = rotation[0]
    yaw = rotation[1]
    roll = rotation[2]
    # print("pitch:",pitch)
    # print("yaw:",yaw)
    # print("roll:",roll)
    # convert_rotation = (math.radians(roll), math.radians(-pitch), math.radians(yaw))
    # the pitch is in CARLA world frame
    # here for camera, it is in the camera frame, pitch is roll
    convert_rotation = (math.radians(90-pitch), math.radians(roll), math.radians(-yaw))
    return convert_rotation

def convert_unreal2carla_in_radians(rotation):
    pitch = rotation[0]
    yaw = rotation[1]
    roll = rotation[2]

    converted_pitch = math.radians(-roll)
    converted_yaw = math.radians(90 - yaw)
    converted_roll = math.radians(90-pitch)
    return (converted_roll, converted_pitch, converted_yaw)
    
if __name__ == "__main__":
    pass