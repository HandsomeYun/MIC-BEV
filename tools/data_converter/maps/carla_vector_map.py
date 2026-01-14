# -*- coding: utf-8 -*-
"""
Code description.
"""
# Author: Zhaoliang Zheng <zhz03@g.ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

import carla
import os
import matplotlib.pyplot as plt
from carla import LaneType

def visualize_carla_map_waypoint():
    # Connect to the CARLA client
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    # Get the current world and map
    world = client.get_world()
    carla_map = world.get_map()

    # get the town name 
    town_name = carla_map.name

    # Extract waypoints and topology information
    waypoints = carla_map.generate_waypoints(distance=2.0)
    topology = carla_map.get_topology()

    # Prepare data containers
    x_coords = []
    y_coords = []

    # Extract the positions of waypoints
    for waypoint in waypoints:
        location = waypoint.transform.location
        x_coords.append(location.x)
        y_coords.append(location.y)

    # Extract the coordinates of topology start and end points
    # for segment in topology:
    #     start = segment[0].transform.location
    #     end = segment[1].transform.location
    #     plt.plot([start.x, end.x], [start.y, end.y], color='blue', linewidth=0.8)

    # Plot the waypoints
    plt.scatter(x_coords, y_coords, s=2, color='red', label='Waypoints')

    # Configure the plot
    plt.title("CARLA Vector Map")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.axis('equal')

    # Display the map
    plt.show()

    # Acquire OpenDRIVE format map
    opendrive_content = carla_map.to_opendrive()

    # get the base path of town_name
    base_name = os.path.basename(town_name)
    print(base_name)

    save_file_name = f"/data2/mcbev-testdata/CARLA_map/{base_name}.xodr"

    # save the map to a file
    with open(save_file_name, "w") as f:
        f.write(opendrive_content)

def visualize_carla_map_new():
    # Connect to the CARLA client
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    # Get the current world and map
    world = client.get_world()
    carla_map = world.get_map()

    # Get the town name
    town_name = carla_map.name

    # Extract waypoints and topology information
    waypoints = carla_map.generate_waypoints(distance=2.0)
    topology = carla_map.get_topology()

    # Prepare data containers by lane type and junction
    waypoint_categories = {
        'Driving': {'x': [], 'y': [], 'color': 'red', 'label': 'Driving Lane'},
        'Sidewalk': {'x': [], 'y': [], 'color': 'green', 'label': 'Sidewalk'},
        'Shoulder': {'x': [], 'y': [], 'color': 'orange', 'label': 'Shoulder'},
        'Biking': {'x': [], 'y': [], 'color': 'purple', 'label': 'Bike Lane'},
        'Parking': {'x': [], 'y': [], 'color': 'brown', 'label': 'Parking Lane'},
        'Border': {'x': [], 'y': [], 'color': 'gray', 'label': 'Border'},
        'Other': {'x': [], 'y': [], 'color': 'pink', 'label': 'Other Lanes'},
        'Junction': {'x': [], 'y': [], 'color': 'blue', 'label': 'Junction Area'}
    }

    # Classify waypoints by lane type and whether in a junction
    for waypoint in waypoints:
        location = waypoint.transform.location
        # Check if waypoint is in a junction
        if waypoint.is_junction:
            waypoint_categories['Junction']['x'].append(location.x)
            waypoint_categories['Junction']['y'].append(location.y)
        else:
            lane_type_str = None
            if waypoint.lane_type == LaneType.Driving:
                lane_type_str = 'Driving'
            elif waypoint.lane_type == LaneType.Sidewalk:
                lane_type_str = 'Sidewalk'
            elif waypoint.lane_type == LaneType.Shoulder:
                lane_type_str = 'Shoulder'
            elif waypoint.lane_type == LaneType.Biking:
                lane_type_str = 'Biking'
            elif waypoint.lane_type == LaneType.Parking:
                lane_type_str = 'Parking'
            elif waypoint.lane_type == LaneType.Border:
                lane_type_str = 'Border'
            else:
                lane_type_str = 'Other'
            
            waypoint_categories[lane_type_str]['x'].append(location.x)
            waypoint_categories[lane_type_str]['y'].append(location.y)

    # Draw topology segments (road skeleton)
    # In the topology, segments are pairs of (start_wp, end_wp)
    for segment in topology:
        start = segment[0].transform.location
        end = segment[1].transform.location
        # Here, we use navy blue lines to represent the main road body
        plt.plot([start.x, end.x], [start.y, end.y], color='navy', linewidth=0.5, alpha=0.5)

    # Plot each category of waypoints
    for cat_name, cat_data in waypoint_categories.items():
        if len(cat_data['x']) > 0:
            plt.scatter(cat_data['x'], cat_data['y'], s=2, color=cat_data['color'], label=cat_data['label'])

    # Configure the plot
    plt.title("CARLA Vector Map - " + town_name)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.axis('equal')

    # Display the map
    plt.show()

    # Acquire OpenDRIVE format map
    opendrive_content = carla_map.to_opendrive()

    # Get the base path of town_name
    base_name = os.path.basename(town_name)
    print(base_name)

    save_file_name = f"./s_metrics/maps/example_data/xodr/{base_name}.xodr"

    # create the directory if it does not exist
    os.makedirs(os.path.dirname(save_file_name), exist_ok=True)

    # Save the map to a file
    with open(save_file_name, "w") as f:
        f.write(opendrive_content)

def visualize_carla_map_waypoint_3d():
    # Connect to the CARLA client
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    # Get the current world and map
    world = client.get_world()
    carla_map = world.get_map()

    # Get the town name
    town_name = carla_map.name

    # Extract waypoints and topology information
    waypoints = carla_map.generate_waypoints(distance=2.0)
    topology = carla_map.get_topology()

    # Prepare data containers
    x_coords = []
    y_coords = []
    z_coords = []

    # Extract the positions of waypoints
    for waypoint in waypoints:
        location = waypoint.transform.location
        x_coords.append(location.x)
        y_coords.append(location.y)
        z_coords.append(location.z)
        print(location.x, location.y, location.z)

    # Plot the waypoints in 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x_coords, y_coords, z_coords, s=2, color='red', label='Waypoints')

    # Extract the coordinates of topology start and end points and plot connections
    # for segment in topology:
    #     start = segment[0].transform.location
    #     end = segment[1].transform.location
    #     ax.plot([start.x, end.x], [start.y, end.y], [start.z, end.z], color='blue', linewidth=0.8)

    # Configure the plot
    ax.set_title("CARLA 3D Vector Map")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()

    set_equal = True
    if set_equal:
        # Set equal aspect ratio for x, y, z axes
        max_range = max(
            max(x_coords) - min(x_coords),
            max(y_coords) - min(y_coords),
            max(z_coords) - min(z_coords)
        ) / 2.0

        mid_x = (max(x_coords) + min(x_coords)) * 0.5
        mid_y = (max(y_coords) + min(y_coords)) * 0.5
        mid_z = (max(z_coords) + min(z_coords)) * 0.5

        # Apply equal aspect ratio
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        # ax.set_zlim(mid_z - max_range, mid_z + max_range)
        ax.set_zlim(-20, 20)
        # print(mid_z - max_range)
        # print(mid_z + max_range)

        ax.set_box_aspect([1, 1, 1])  # Set the aspect ratio of the box to equal for all axes

    # Display the map
    plt.show()

def main():
    visualize_carla_map_new()

def test2():
    visualize_carla_map_waypoint_3d()

if __name__ == "__main__":
    # visualize_carla_map_waypoint()
    main()
    
    # Call the function to visualize waypoint in 3d 
    # test2()