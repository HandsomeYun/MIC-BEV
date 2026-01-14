#!/usr/bin/env python

# Copyright (c) 2018 Intel Labs.
# authors: German Ros (german.ros@intel.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

""" Module with auxiliary functions. """

import math
import importlib
import random
import numpy as np


def get_speed(vehicle, meters=False):
    """
    Compute speed of a vehicle in Km/h.

    Parameters
    ----------
    meters : bool
        Whether to use m/s (True) or km/h (False).

    vehicle : carla.vehicle
        The vehicle for which speed is calculated.

    Returns
    -------
    speed : float
        The vehicle speed.
    """
    vel = vehicle.get_velocity()
    vel_meter_per_second = math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)
    return vel_meter_per_second if meters else 3.6 * vel_meter_per_second


def get_acc(vehicle, meters=False):
    """
    Compute acceleration of a vehicle.

    Parameters
    ----------
    meters : bool
        Whether to use m/s^2 (True) or km/h^2 (False).

    vehicle : carla.vehicle
        The vehicle for which speed is calculated.

    Returns
    -------
    acceleration : float
        The vehicle speed.
    """
    acc = vehicle.get_acceleration()
    acc_meter_per_second = math.sqrt(acc.x ** 2 + acc.y ** 2 + acc.z ** 2)

    return acc_meter_per_second if meters else 3.6 * acc_meter_per_second


def cal_distance_angle(target_location, current_location, orientation):
    """
    Calculate the vehicle current relative distance to target location.

    Parameters
    ----------
    target_location : carla.Location
        The target location.

    current_location : carla.Location
        The current location .

    orientation : carla.Rotation
        Orientation of the reference object.

    Returns
    -------
    distance : float
        The measured distance from current location to target location.

    d_angle : float)
        The measured rotation (angle) froM current location
        to target location.
    """
    target_vector = np.array([target_location.x -
                              current_location.x, target_location.y -
                              current_location.y])
    norm_target = np.linalg.norm(target_vector) + 1e-10

    forward_vector = np.array(
        [math.cos(math.radians(orientation)),
         math.sin(math.radians(orientation))])
    d_angle = math.degrees(
        math.acos(
            np.clip(
                np.dot(
                    forward_vector, target_vector) / norm_target, -1., 1.)))

    return norm_target, d_angle


def distance_vehicle(waypoint, vehicle_transform):
    """
    Returns the 2D distance from a waypoint to a vehicle

    Parameters
    ----------
    waypoint : carla.Waypoint
        Actual waypoint.

    vehicle_transform : carla.transform
        Transform of the target vehicle.
    """
    loc = vehicle_transform.location
    if hasattr(waypoint, 'is_junction'):
        x = waypoint.transform.location.x - loc.x
        y = waypoint.transform.location.y - loc.y
    else:
        x = waypoint.location.x - loc.x
        y = waypoint.location.y - loc.y

    return math.sqrt(x * x + y * y)


def vector(location_1, location_2):
    """
    Returns the unit vector from location_1 to location_2.

    Parameters
    ----------
    location_1 : carla.location
        Start location of the vector.

    location_2 : carla.location
        End location of the vector.
    """
    x = location_2.x - location_1.x
    y = location_2.y - location_1.y
    z = location_2.z - location_1.z
    norm = np.linalg.norm([x, y, z]) + np.finfo(float).eps

    return [x / norm, y / norm, z / norm]


def compute_distance(location_1, location_2):
    """
    Euclidean distance between 3D points.

    Parameters
    ----------
    location_1 : carla.Location
        Start point of the measurement.

    location_2 : carla.Location
        End point of the measurement.
    """
    x = location_2.x - location_1.x
    y = location_2.y - location_1.y
    z = location_2.z - location_1.z
    norm = np.linalg.norm([x, y, z]) + np.finfo(float).eps
    return norm


def positive(num):
    """
    Return the given number if positive, else 0
    """
    return num if num > 0.0 else 0.0


def get_speed_sumo(sumo2carla_ids, carla_id):
    """
    Get the speed of the vehicles controlled by sumo.

    Parameters
    ----------
    sumo2carla_ids : dict
        Sumo-carla mapping dictionary.

    carla_id : int
        Carla actor id.

    Returns
    -------
    speed : float
        The speed retrieved from the sumo server, -1 if the carla_id not
        found.
    """
    # python will only import this once and then save it in cache. so the
    # efficiency won't affected during the loop.
    traci = importlib.import_module("traci")

    for key, value in sumo2carla_ids.items():
        if int(value) == carla_id:
            vehicle_speed = traci.vehicle.getSpeed(key)
            return vehicle_speed

    return -1

def output_intersection_range(intersection_center, range_x, range_y):
    """
    Calculate the intersection range based on the given center and ranges.
    Parameters:
    intersection_center (list): The center coordinates of the intersection.
    range_x (float): The range in the x-axis.
    range_y (float): The range in the y-axis.
    Returns:
    tuple: A tuple containing the minimum and maximum values for x and y coordinates of the intersection range.
    """

    # This is a square shape model
    intersection_center[1] = - intersection_center[1]
    x_min = intersection_center[0] - range_x
    x_max = intersection_center[0] + range_x
    y_min = intersection_center[1] - range_y
    y_max = intersection_center[1] + range_y
    return x_min, x_max, y_min, y_max

def get_random_color():
    """
    Generate a random color in hexadecimal format.
    Returns:
        str: A string representing a random color in the format '#RRGGBB'.
    """

    return "#{:06x}".format(random.randint(0, 0xFFFFFF))

def rotation_matrix_from_euler(roll, pitch, yaw):
    """Compute rotation matrix from roll, pitch, yaw angles."""
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(roll), -math.sin(roll)],
                    [0, math.sin(roll), math.cos(roll)]])
    
    R_y = np.array([[math.cos(pitch), 0, math.sin(pitch)],
                    [0, 1, 0],
                    [-math.sin(pitch), 0, math.cos(pitch)]])
    
    R_z = np.array([[math.cos(yaw), -math.sin(yaw), 0],
                    [math.sin(yaw),  math.cos(yaw),  0],
                    [0,              0,              1]])
    
    # return R_z @ R_y @ R_x 
    return R_z @ R_y @ R_x # # Apply rotations in the order: first roll, then pitch, then yaw.