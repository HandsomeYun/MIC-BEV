"""

"""


import os
import math
from lxml import etree
from tqdm import tqdm
import xml.etree.ElementTree as ET
from tools.data_converter.maps.xodr_parser.opendriveparser import parse_opendrive
from math import pi, sin, cos
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pickle 
from tools.data_converter.maps.xodr_parser.opendriveparser.misc import output_intersection_range
from tools.data_converter.maps.utils import to_color, XODR_FILE, DRIVING_COLOR, TYPE_COLOR_DICT, \
                                            COLOR_CENTER_LANE, COLOR_REFERECE_LINE
import argparse

# Prepare sample step.
STEP = 0.1

# Areas of intersection at Tintersection_Town05 that needs to be cut.
CUT_AREAS = [[-258.1,-222.1,-43.7,-12.7],[-256.9,-222.6,9.0,38.4],[-300,-280.9,-20.9,24.35]] 
# area 1:
# x = -256.7 , y = -12.4
# x = -222.1, y = -43.7
# area 2:
# x = -256.9, y = 9.0
# x = -222.6, y = 38.4
# area 3:
# x = -280.7, 21.2 
# x = -300, y = -20.9

def load_xodr_and_parse(file=XODR_FILE):
    """
    Load and parse .xodr file.
    :param file:
    :return:
    """
    with open(file, 'r') as fh:
        parser = etree.XMLParser()
        root_node = etree.parse(fh, parser).getroot()
        road_network = parse_opendrive(root_node)
    return road_network

def load_xodr_and_parse1(file=XODR_FILE):
    """
    Load and parse .xodr file, extracting roads and crosswalk objects.
    """
    with open(file, 'r') as fh:
        parser = etree.XMLParser()
        root_node = etree.parse(fh, parser).getroot()
        road_network = parse_opendrive(root_node)

        # Extract crosswalks from objects
        crosswalks = []
        objects = root_node.findall('.//object[@type="crosswalk"]')
        for obj in objects:
            outline = obj.find('outline')
            corners = []
            for corner in outline.findall('cornerLocal'):
                u = float(corner.attrib['u'])
                v = float(corner.attrib['v'])
                corners.append({'u': u, 'v': v})

            crosswalks.append({
                'id': obj.attrib['id'],
                'name': obj.attrib.get('name', 'Crosswalk'),
                'outline': corners
            })
    print(f"Found {len(crosswalks)} crosswalks")
    return road_network, crosswalks

def parse_road_geometries(road_elem):
    geometries = []
    planView = road_elem.find('planView')
    for geom in planView.findall('geometry'):
        s = float(geom.get('s'))
        x = float(geom.get('x'))
        y = float(geom.get('y'))
        hdg = float(geom.get('hdg'))
        length = float(geom.get('length'))

        line_elem = geom.find('line')
        arc_elem = geom.find('arc')
        if line_elem is not None:
            geom_type = 'line'
            curvature = 0.0
        elif arc_elem is not None:
            geom_type = 'arc'
            curvature = float(arc_elem.get('curvature'))
        else:
            raise ValueError("Unsupported geometry type")

        geometries.append((s, x, y, hdg, length, geom_type, curvature))
    return geometries

def eval_geometry(geometries, s):
    """
    Locate the geometry segment based on the given 's' value and compute the global coordinates
    and heading on the reference line.

    Parameters:
    - geometries: List of geometry segments, each defined by (gs, gx, gy, ghdg, glen, gtype, gcurv)
      where:
        gs    : Starting position along the reference line
        gx, gy: Global X and Y coordinates of the geometry's starting point
        ghdg  : Heading angle of the geometry in radians
        glen  : Length of the geometry segment
        gtype : Type of geometry ('line' or 'arc')
        gcurv : Curvature of the geometry (only for arcs)
    - s: Position along the reference line where the evaluation is to be performed

    Returns:
    - Tuple of (X, Y, HDG) representing the global coordinates and heading at position 's'
    """
    for (gs, gx, gy, ghdg, glen, gtype, gcurv) in geometries:
        # Check if 's' falls within the current geometry segment
        if gs <= s <= gs + glen:
            ds = s - gs  # Distance from the start of the geometry segment

            if gtype == 'line':
                # For straight line segments, calculate X and Y using trigonometry
                X = gx + ds * math.cos(ghdg)
                Y = gy + ds * math.sin(ghdg)
                HDG = ghdg  # Heading remains the same as the geometry's heading
            else:
                # For arc segments, calculate based on curvature
                R = 1.0 / gcurv  # Radius of the arc
                HDG = ghdg + ds * gcurv  # Updated heading after traversing the arc
                # Calculate the new X and Y using the arc formula
                X = gx - R * math.sin(ghdg) + R * math.sin(HDG)
                Y = gy + R * math.cos(ghdg) - R * math.cos(HDG)

            return X, Y, HDG  # Return the computed global coordinates and heading

    # If 's' exceeds the last geometry segment, use the last segment to compute the position
    last = geometries[-1]
    gs, gx, gy, ghdg, glen, gtype, gcurv = last
    ds = glen  # Use the full length of the last geometry segment

    if gtype == 'line':
        # Calculate position for the last line segment
        X = gx + ds * math.cos(ghdg)
        Y = gy + ds * math.sin(ghdg)
        HDG = ghdg
    else:
        # Calculate position for the last arc segment
        R = 1.0 / gcurv
        HDG = ghdg + ds * gcurv
        X = gx - R * math.sin(ghdg) + R * math.sin(HDG)
        Y = gy + R * math.cos(ghdg) - R * math.cos(HDG)

    return X, Y, HDG  # Return the computed global coordinates and heading

def convert_object_to_world(geometries, s_obj, t_obj, hdg_obj, corners):
    """
    Convert the local coordinates of an object (e.g., a crosswalk) to global world coordinates.

    Parameters:
    - geometries: List of geometry segments as defined in eval_geometry
    - s_obj: Longitudinal position of the object along the reference line
    - t_obj: Lateral offset of the object from the reference line
    - hdg_obj: Heading of the object relative to the reference line
    - corners: List of tuples representing the object's outline in local coordinates (u, v)

    Returns:
    - List of dictionaries containing the global X and Y coordinates of the object's outline points
    """
    # Get the reference line's global position and heading at s_obj
    x_ref, y_ref, hdg_ref = eval_geometry(geometries, s_obj)

    # Offset the reference position by t_obj along the normal direction
    # In OpenDRIVE, positive 't' is to the left of the reference line, so the offset direction is (hdg_ref + pi/2)
    x0 = x_ref + t_obj * math.cos(hdg_ref + math.pi / 2)
    y0 = y_ref + t_obj * math.sin(hdg_ref + math.pi / 2)

    # Compute the global heading by adding the object's heading to the reference heading
    HDG = hdg_ref + hdg_obj

    world_points = []
    for (u, v) in corners:
        # Convert each local corner point (u, v) to global coordinates
        # 'u' is along the heading direction, 'v' is perpendicular to it
        Xw = x0 + u * math.cos(HDG) - v * math.sin(HDG)
        Yw = y0 + u * math.sin(HDG) + v * math.cos(HDG)
        world_points.append({'X': Xw, 'Y': Yw})  # Store the global coordinates

    return world_points

def parse_crosswalks(file=XODR_FILE):
    """
    Parse crosswalk objects from an OpenDRIVE (XODR) file and convert their local coordinates
    to global world coordinates.

    Parameters:
    - file: Path to the XODR file to be parsed

    Returns:
    - List of dictionaries, each containing the crosswalk ID and its outline in global coordinates
    """
    # Parse the XML content of the XODR file
    tree = ET.parse(file)
    root = tree.getroot()

    crosswalks_global = []

    # Iterate over all 'road' elements in the XODR file
    for road in root.findall('road'):
        # Parse the reference line geometries for the current road
        geometries = parse_road_geometries(road)  # Assume this function is defined elsewhere

        objects_elem = road.find('objects')  # Find the 'objects' section within the road
        if objects_elem is None:
            continue  # Skip if there are no objects

        # Iterate over all 'object' elements within the 'objects' section
        for obj in objects_elem.findall('object'):
            if obj.get('type') == 'crosswalk':
                # Extract object attributes with default values if not present
                s_obj = float(obj.get('s', '0'))
                t_obj = float(obj.get('t', '0'))
                hdg_obj = float(obj.get('hdg', '0'))

                outline = obj.find('outline')  # Find the 'outline' element defining the crosswalk's shape
                corners = []
                for c in outline.findall('cornerLocal'):
                    # Extract local corner coordinates
                    u = float(c.get('u', '0'))
                    v = float(c.get('v', '0'))
                    corners.append((u, v))

                # Convert the crosswalk's local coordinates to global world coordinates
                world_points = convert_object_to_world(geometries, s_obj, t_obj, hdg_obj, corners)
                crosswalks_global.append({
                    'id': obj.get('id', 'N/A'),  # Use 'N/A' if the ID is not provided
                    'outline': world_points
                })

    return crosswalks_global

def calculate_reference_points_of_one_geometry(geometry, length, step=0.01):
    """
    Calculate the stepwise reference points with position(x, y), tangent and distance between the point and the start.
    :param geometry:
    :param length:
    :param step:
    :return:
    """
    nums = int(length / step)
    res = []
    for i in range(nums):
        s_ = step * i
        pos_, tangent_ = geometry.calcPosition(s_)
        x, y = pos_
        one_point = {
            "position": (x, y),     # The location of the reference point
            "tangent": tangent_,    # Orientation of the reference point
            "s_geometry": s_,       # The distance between the start point of the geometry and current point along the reference line
        }
        res.append(one_point)
    return res

def get_geometry_length(geometry):
    """
    Get the length of one geometry (or the length of the reference line of the geometry).
    :param geometry:
    :return:
    """
    if hasattr(geometry, "length"):
        length = geometry.length
    elif hasattr(geometry, "_length"):
        length = geometry._length           # Some geometry has the attribute "_length".
    else:
        raise AttributeError("No attribute length found!!!")
    return length

def get_all_reference_points_of_one_road(geometries, step=0.01):
    """
    Obtain the sampling point of the reference line of the road, including:
    the position of the point
    the direction of the reference line at the point
    the distance of the point along the reference line relative to the start of the road
    the distance of the point relative to the start of geometry along the reference line
    :param geometries: Geometries of one road.
    :param step: Calculate steps.
    :return:
    """
    reference_points = []
    s_start_road = 0
    for geometry_id, geometry in enumerate(geometries):
        geometry_length = get_geometry_length(geometry)

        # Calculate all the reference points of current geometry.
        pos_tangent_s_list = calculate_reference_points_of_one_geometry(geometry, geometry_length, step=step)

        # As for every reference points, add the distance start by road and its geometry index.
        pos_tangent_s_s_list = [{**point,
                                 "s_road": point["s_geometry"]+s_start_road,
                                 "index_geometry": geometry_id}
                                for point in pos_tangent_s_list]
        reference_points.extend(pos_tangent_s_s_list)

        s_start_road += geometry_length
    return reference_points

def get_width(widths, s):
    assert isinstance(widths, list), TypeError(type(widths))
    widths.sort(key=lambda x: x.sOffset)
    current_width = None
    # EPS = 1e-5
    milestones = [width.sOffset for width in widths] + [float("inf")]

    control_mini_section = [(start, end) for (start, end) in zip(milestones[:-1], milestones[1:])]
    for width, start_end in zip(widths, control_mini_section):
        start, end = start_end
        if start <= s < end:
            ds = s - width.sOffset
            current_width = width.a + width.b * ds + width.c * ds ** 2 + width.d * ds ** 3
    return current_width

def get_lane_offset(lane_offsets, section_s, length=float("inf")):

    assert isinstance(lane_offsets, list), TypeError(type(lane_offsets))
    if not lane_offsets:
        return 0
    lane_offsets.sort(key=lambda x: x.sPos)
    current_offset = 0
    EPS = 1e-5
    milestones = [lane_offset.sPos for lane_offset in lane_offsets] + [length+EPS]

    control_mini_section = [(start, end) for (start, end) in zip(milestones[:-1], milestones[1:])]
    for offset_params, start_end in zip(lane_offsets, control_mini_section):
        start, end = start_end
        if start <= section_s < end:
            ds = section_s - offset_params.sPos
            current_offset = offset_params.a + offset_params.b * ds + offset_params.c * ds ** 2 + offset_params.d * ds ** 3
    return current_offset

class LaneOffsetCalculate:

    def __init__(self, lane_offsets):
        lane_offsets = list(sorted(lane_offsets, key=lambda x: x.sPos))
        lane_offsets_dict = dict()
        for lane_offset in lane_offsets:
            a = lane_offset.a
            b = lane_offset.b
            c = lane_offset.c
            d = lane_offset.d
            s_start = lane_offset.sPos
            lane_offsets_dict[s_start] = (a, b, c, d)
        self.lane_offsets_dict = lane_offsets_dict

    def calculate_offset(self, s):
        for s_start, (a, b, c, d) in reversed(self.lane_offsets_dict.items()): # e.g. 75, 25
            if s >= s_start:
                ds = s - s_start
                offset = a + b * ds + c * ds ** 2 + d * ds ** 3
                return offset
        return 0


def calculate_area_of_one_left_lane(left_lane, points, most_left_points):
    inner_points = most_left_points[:]

    widths = left_lane.widths
    update_points = []
    for reference_point, inner_point in zip(points, inner_points):

        tangent = reference_point["tangent"]
        s_lane_section = reference_point["s_lane_section"]
        lane_width = get_width(widths, s_lane_section)

        normal_left = tangent + pi / 2
        x_inner, y_inner = inner_point

        lane_width_offset = lane_width

        x_outer = x_inner + cos(normal_left) * lane_width_offset
        y_outer = y_inner + sin(normal_left) * lane_width_offset

        update_points.append((x_outer, y_outer))

    outer_points = update_points[:]
    most_left_points = outer_points[:]

    current_ara = {
        "inner": inner_points,
        "outer": outer_points,
    }
    return current_ara, most_left_points

def calculate_area_of_one_right_lane(right_lane, points, most_right_points):
    inner_points = most_right_points[:]

    widths = right_lane.widths
    update_points = []
    for reference_point, inner_point in zip(points, inner_points):

        tangent = reference_point["tangent"]
        s_lane_section = reference_point["s_lane_section"]
        lane_width = get_width(widths, s_lane_section)

        normal_eight = tangent - pi / 2
        x_inner, y_inner = inner_point

        lane_width_offset = lane_width

        x_outer = x_inner + cos(normal_eight) * lane_width_offset
        y_outer = y_inner + sin(normal_eight) * lane_width_offset

        update_points.append((x_outer, y_outer))

    outer_points = update_points[:]
    most_right_points = outer_points[:]

    current_ara = {
        "inner": inner_points,
        "outer": outer_points,
    }
    return current_ara, most_right_points

def calculate_lane_area_within_one_lane_section(lane_section, points):
    """
    Lane areas are represented by boundary lattice. Calculate boundary points of every lanes.
    :param lane_section:
    :param points:
    :return:
    """

    all_lanes = lane_section.allLanes

    # Process the lane indexes.
    left_lanes = [lane for lane in all_lanes if int(lane.id) > 0]
    right_lanes = [lane for lane in all_lanes if int(lane.id) < 0]
    left_lanes.sort(key=lambda x: x.id)
    right_lanes.sort(reverse=True, key=lambda x: x.id)

    # Get the lane area of left lanes and the most left lane line.
    left_lanes_area = dict()
    most_left_points = [point["position_center_lane"] for point in points][:]
    for left_lane in left_lanes:
        current_area, most_left_points = calculate_area_of_one_left_lane(left_lane, points, most_left_points)
        left_lanes_area[left_lane.id] = current_area

    # Get the lane area of right lanes and the most right lane line.
    right_lanes_area = dict()
    most_right_points = [point["position_center_lane"] for point in points][:]
    for right_lane in right_lanes:
        current_area, most_right_points = calculate_area_of_one_right_lane(right_lane, points, most_right_points)
        right_lanes_area[right_lane.id] = current_area

    return left_lanes_area, right_lanes_area, most_left_points, most_right_points

def calculate_points_of_reference_line_of_one_section(points):
    """
    Calculate center lane points accoding to the reference points and offsets.
    :param points: Points on reference line including position and tangent.
    :return: Updated points.
    """
    res = []
    for point in points:
        tangent = point["tangent"]
        x, y = point["position"]    # Points on reference line.
        normal = tangent + pi / 2
        lane_offset = point["lane_offset"]  # Offset of center lane.

        x += cos(normal) * lane_offset
        y += sin(normal) * lane_offset

        point = {
            **point,
            "position_center_lane": (x, y),
        }
        res.append(point)
    return res

def calculate_s_lane_section(reference_points, lane_sections):

    res = []
    for point in reference_points:

        for lane_section in reversed(lane_sections):
            if point["s_road"] >= lane_section.sPos:
                res.append(
                    {
                        **point,
                        "s_lane_section": point["s_road"] - lane_section.sPos,
                        "index_lane_section": lane_section.idx,
                    }
                )
                break
    return res

def uncompress_dict_list(dict_list: list):
    assert isinstance(dict_list, list), TypeError("Keys")
    if not dict_list:
        return dict()

    keys = set(dict_list[0].keys())
    for dct in dict_list:
        cur = set(dct.keys())
        assert keys == cur, "Inconsistency of dict keys! {} {}".format(keys, cur)

    res = dict()
    for sample in dict_list:
        for k, v in sample.items():
            if k not in res:
                res[k] = [v]
            else:
                res[k].append(v)

    keys = list(sorted(list(keys)))
    res = {k: res[k] for k in keys}
    return res

def get_lane_line(section_data: dict):
    """
    Extract lane lines dividing lanes.
    :param section_data:
    :return:
    """
    left_lanes_area = section_data["left_lanes_area"]
    right_lanes_area = section_data["right_lanes_area"]

    lane_line_left = dict()
    if left_lanes_area:
        indexes = list(left_lanes_area.keys())  # 默认是排好序的
        for index_inner, index_outer in zip(indexes, indexes[1:] + ["NAN"]):
            lane_line_left[(index_inner, index_outer)] = left_lanes_area[index_inner]["outer"]

    lane_line_right = dict()
    if right_lanes_area:
        indexes = list(right_lanes_area.keys())  # 默认是排好序的
        for index_inner, index_outer in zip(indexes, indexes[1:] + ["NAN"]):
            lane_line_right[(index_inner, index_outer)] = right_lanes_area[index_inner]["outer"]

    return {"lane_line_left": lane_line_left, "lane_line_right": lane_line_right}

def get_lane_area_of_one_road(road, step=0.01):
    """
    Get all corresponding positions of every lane section in one road.
    :param road:
    :param step:
    :return: A dictionary of dictionary: {(road id, lane section id): section data}
    Section data is a dictionary of position information.
    section_data = {
        "left_lanes_area": left_lanes_area,
        "right_lanes_area": right_lanes_area,
        "most_left_points": most_left_points,
        "most_right_points": most_right_points,
        "types": types,
        "reference_points": uncompressed_lane_section_data,
    }
    """
    geometries = road.planView._geometries
    # Lane offset is the offset between center lane (width is 0) and the reference line.
    lane_offsets = road.lanes.laneOffsets
    lane_offset_calculate = LaneOffsetCalculate(lane_offsets=lane_offsets)
    lane_sections = road.lanes.laneSections
    lane_sections = list(sorted(lane_sections, key=lambda x: x.sPos))   # Sort the lane sections by start position.

    reference_points = get_all_reference_points_of_one_road(geometries, step=step)  # Extract the reference points.

    # Calculate the offsets of center lane.
    reference_points = [{**point, "lane_offset":  lane_offset_calculate.calculate_offset(point["s_road"])}
                        for point in reference_points]

    # Calculate the points of center lane based on reference points and offsets.
    reference_points = calculate_points_of_reference_line_of_one_section(reference_points)

    # Calculate the distance of each point starting from the current section along the direction of the reference line.
    reference_points = calculate_s_lane_section(reference_points, lane_sections)

    total_areas = dict()
    for lane_section in lane_sections:
        section_start = lane_section.sPos  # Start position of the section in current road.
        section_end = lane_section.sPos + lane_section.length  # End position of the section in current road.

        # Filter out the points belonging to current lane section.
        current_reference_points = list(filter(lambda x: section_start <= x["s_road"] < section_end, reference_points))

        # Calculate the boundary point of every lane in current lane section.
        area = calculate_lane_area_within_one_lane_section(lane_section, current_reference_points)
        left_lanes_area, right_lanes_area, most_left_points, most_right_points = area

        # Extract types and indexes.
        types = {lane.id: lane.type for lane in lane_section.allLanes if lane.id != 0}
        index = (road.id, lane_section.idx)

        # Convert dict list to list dict of the reference points information.
        uncompressed_lane_section_data = uncompress_dict_list(current_reference_points)

        # Integrate all the information of current lane section of current road.
        section_data = {
            "left_lanes_area": left_lanes_area,
            "right_lanes_area": right_lanes_area,
            "most_left_points": most_left_points,
            "most_right_points": most_right_points,
            "types": types,
            "reference_points": uncompressed_lane_section_data,  # 这些是lane section的信息
        }

        # Get all lane lines with their left and right lanes.
        lane_line = get_lane_line(section_data)
        section_data.update(lane_line)

        total_areas[index] = section_data

    return total_areas

def get_all_lanes(road_network, step=0.1):
    """
    Get all lanes of one road network.
    :param road_network: Parsed road network.
    :param step: Step of calculation.
    :return: Dictionary with the following format:
        keys: (road id, lane section id)
        values: dict(left_lanes_area, right_lanes_area, most_left_points, most_right_points, types, reference_points)
    """
    roads = road_network.roads
    total_areas_all_roads = dict()

    for road in tqdm(roads, desc="Calculating boundary points."):
        lanes_of_one_road = get_lane_area_of_one_road(road, step=step)
        total_areas_all_roads = {**total_areas_all_roads, **lanes_of_one_road}

    return total_areas_all_roads

def rescale_color(hex_color, rate=0.5):
    """
    Half the light of input color, e.g. white => grey.
    :param hex_color: e.g. #a55f13
    :param rate: Scale rate from 0 to 1.
    :return:
    """

    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)

    # Half the colors.
    r = min(255, max(0, int(r * rate)))
    g = min(255, max(0, int(g * rate)))
    b = min(255, max(0, int(b * rate)))

    r = hex(r)[2:]
    g = hex(g)[2:]
    b = hex(b)[2:]

    r = r.rjust(2, "0")
    g = g.rjust(2, "0")
    b = b.rjust(2, "0")

    new_hex_color = '#{}{}{}'.format(r, g, b)
    return new_hex_color

def plot_planes_of_roads(total_areas,save_folder="",save_pkl=False):
    """
    Plot the roads.
    :param total_areas:
    :param save_folder:
    :return:
    """
    plot_control_center_n_ref_line = True
    plot_boundary_line = True
    plot_lane_area = True
    # save_pkl = True
    
    if save_pkl:
        # judge if the save_folder exists
        if not os.path.exists(save_folder):
            raise FileNotFoundError("The save_folder does not exist, please specify a valid folder.")
        # Ensure the save_folder exists; create it if it doesn't
        os.makedirs(save_folder, exist_ok=True)
        print("save_folder:", save_folder)

    # plt.cla() # Remove plt.cla() and avoid creating multiple figures

    plt.figure(figsize=(160, 90))
    area_select = 10  # select one from 10 boundary points for accelerating.

    all_types = set()

    # Plot lane area.
    if plot_lane_area:
        for k, v in tqdm(total_areas.items(), desc="Ploting Roads"):
            
            left_lanes_area = v["left_lanes_area"]
            right_lanes_area = v["right_lanes_area"]

            types = v["types"]

            for left_lane_id, left_lane_area in left_lanes_area.items():
                type_of_lane = types[left_lane_id]
                all_types.add(type_of_lane)
                lane_color = TYPE_COLOR_DICT[type_of_lane]
                inner_points = left_lane_area["inner"]
                outer_points = left_lane_area["outer"]

                points_of_one_road = inner_points + outer_points[::-1]
                xs = [i for i, _ in points_of_one_road]
                ys = [i for _, i in points_of_one_road]
                plt.fill(xs, ys, color=lane_color, label=type_of_lane)
                plt.scatter(xs[::area_select], ys[::area_select], color=rescale_color(lane_color, 0.5), s=1)

            for right_lane_id, right_lane_area in right_lanes_area.items():
                type_of_lane = types[right_lane_id]
                all_types.add(type_of_lane)

                lane_color = TYPE_COLOR_DICT[type_of_lane]
                inner_points = right_lane_area["inner"]
                outer_points = right_lane_area["outer"]

                points_of_one_road = inner_points + outer_points[::-1]
                xs = [i for i, _ in points_of_one_road]
                ys = [i for _, i in points_of_one_road]
                plt.fill(xs, ys, color=lane_color, label=type_of_lane)
                plt.scatter(xs[::area_select], ys[::area_select], color=rescale_color(lane_color, 0.5), s=1)

    # Plot boundaries
    if plot_boundary_line:
        for k, v in tqdm(total_areas.items(), desc="Ploting Edges"):
            left_lanes_area = v["left_lanes_area"]
            right_lanes_area = v["right_lanes_area"]

            types = v["types"]
            for left_lane_id, left_lane_area in left_lanes_area.items():
                type_of_lane = types[left_lane_id]
                all_types.add(type_of_lane)
                lane_color = TYPE_COLOR_DICT[type_of_lane]
                inner_points = left_lane_area["inner"]
                outer_points = left_lane_area["outer"]
                points_of_one_road = inner_points + outer_points[::-1]
                xs = [i for i, _ in points_of_one_road]
                ys = [i for _, i in points_of_one_road]
                plt.scatter(xs[::area_select], ys[::area_select], color=rescale_color(lane_color, 0.5), s=1)

            for right_lane_id, right_lane_area in right_lanes_area.items():
                type_of_lane = types[right_lane_id]
                all_types.add(type_of_lane)
                lane_color = TYPE_COLOR_DICT[type_of_lane]
                inner_points = right_lane_area["inner"]
                outer_points = right_lane_area["outer"]
                points_of_one_road = inner_points + outer_points[::-1]
                xs = [i for i, _ in points_of_one_road]
                ys = [i for _, i in points_of_one_road]
                plt.scatter(xs[::area_select], ys[::area_select], color=rescale_color(lane_color, 0.5), s=1)

    # Plot center lane and reference line.
    if plot_control_center_n_ref_line:
        saved_ceter_lanes = dict()
        for k, v in tqdm(total_areas.items(), desc="Ploting Reference and center"):

            reference_points = v["reference_points"]
            if not reference_points:
                continue
            position_reference_points = reference_points["position"]
            position_center_lane = reference_points["position_center_lane"]

            position_reference_points_xs = [x for x, y in position_reference_points]
            position_reference_points_ys = [y for x, y in position_reference_points]
            position_center_lane_xs = [x for x, y in position_center_lane]
            position_center_lane_ys = [y for x, y in position_center_lane]

            saved_ceter_lanes[k] = position_center_lane
            plt.scatter(position_reference_points_xs, position_reference_points_ys, color=COLOR_REFERECE_LINE, s=3)
            plt.scatter(position_center_lane_xs, position_center_lane_ys, color=COLOR_CENTER_LANE, s=2)

    # Create legend.
    legend_dict = {
        # k: Patch(facecolor=v, edgecolor=v, alpha=0.3) for k, v in type_color_dict.items() if k in all_types
        k: Patch(facecolor=v, edgecolor=v, alpha=1.0) for k, v in TYPE_COLOR_DICT.items() if k in all_types
    }
    legend_dict.update({
        "center_lane": Patch(facecolor=COLOR_CENTER_LANE, edgecolor=COLOR_CENTER_LANE, alpha=1.0),
    })
    legend_dict.update({
        "reference_line": Patch(facecolor=COLOR_REFERECE_LINE, edgecolor=COLOR_REFERECE_LINE, alpha=1.0),
    })

    # add crosswalks to legend_dict 
    legend_dict.update({
        "crosswalk": Patch(facecolor=TYPE_COLOR_DICT['crosswalk'], edgecolor=TYPE_COLOR_DICT['crosswalk'], alpha=1.0),
    })    
    # print(legend_dict['crosswalk'])
    crosswalks = parse_crosswalks(XODR_FILE)
    plot_crosswalks(crosswalks, plt.gca())

    # Save the filtered_total_areas to a pickle file
    if save_pkl:
        pickle_path = os.path.join(save_folder, "total_areas.pkl")
        with open(pickle_path, "wb") as f:
            pickle.dump(total_areas, f)
        print(f"Filtered total areas have been saved to {pickle_path}")

    if save_pkl:
        pickle_path2 = os.path.join(save_folder, "crosswalks.pkl")
        with open(pickle_path2, "wb") as f:
            pickle.dump(crosswalks, f)
        print(f"Filtered crosswalks have been saved to {pickle_path2}")
        
    plt.legend(handles=legend_dict.values(), labels=legend_dict.keys(), fontsize=8, loc='upper right')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.show()


def output_intersection_range_inv_y(intersection_center, range_x, range_y):
    # This is a square shape model
    
    intersection_center[1] = - intersection_center[1]
    x_min = intersection_center[0] - range_x
    x_max = intersection_center[0] + range_x
    y_min = intersection_center[1] - range_y
    y_max = intersection_center[1] + range_y
    return x_min, x_max, y_min, y_max

def filter_total_areas(total_areas, x_min=-40, x_max=40, 
                       y_min=-30, y_max=30,
                       z_min=-1,z_max=2):
    """
    Filters the xs and ys in total_areas based on the specified x and y ranges.

    :param total_areas: Dictionary containing road area data.
    :param x_min: Minimum x-coordinate value.
    :param x_max: Maximum x-coordinate value.
    :param y_min: Minimum y-coordinate value.
    :param y_max: Maximum y-coordinate value.
    :return: Filtered total_areas dictionary.
    """
    filtered_areas = {}
    
    for area_key, area_value in total_areas.items():
        filtered_area = {
            "left_lanes_area": {},
            "right_lanes_area": {},
            "types": area_value["types"]
        }
        
        # Process left lanes
        for left_lane_id, left_lane_area in area_value["left_lanes_area"].items():
            # Filter inner points
            filtered_inner = [
                point for point in left_lane_area["inner"]
                if x_min < point[0] < x_max and y_min < point[1] < y_max # and z_min < point[2] < z_max
            ]
            # Filter outer points
            filtered_outer = [
                point for point in left_lane_area["outer"]
                if x_min < point[0] < x_max and y_min < point[1] < y_max # and z_min < point[2] < z_max
            ]
            # Update the filtered area
            filtered_area["left_lanes_area"][left_lane_id] = {
                "inner": filtered_inner,
                "outer": filtered_outer
            }
        
        # Process right lanes
        for right_lane_id, right_lane_area in area_value["right_lanes_area"].items():
            # Filter inner points
            filtered_inner = [
                point for point in right_lane_area["inner"]
                if x_min < point[0] < x_max and y_min < point[1] < y_max # and z_min < point[2] < z_max
            ]
            # Filter outer points
            filtered_outer = [
                point for point in right_lane_area["outer"]
                if x_min < point[0] < x_max and y_min < point[1] < y_max # and z_min < point[2] < z_max
            ]
            # Update the filtered area
            filtered_area["right_lanes_area"][right_lane_id] = {
                "inner": filtered_inner,
                "outer": filtered_outer
            }
        
        filtered_areas[area_key] = filtered_area
    
    return filtered_areas

def filter_total_areas_cut_areas(total_areas, cut_areas = []):  
    for cut_area in cut_areas:
        total_areas = filter_total_areas_cut_area(total_areas, cut_area = cut_area)
    return total_areas  

def filter_total_areas_cut_area(total_areas, cut_area = []):
    """
    Filters the xs and ys in total_areas based on the specified x and y ranges.

    :param total_areas: Dictionary containing road area data.
    :cut_areas: [[x_min,x_max,y_min,y_max],[x_min,x_max,y_min,y_max]].
    :return: Filtered total_areas dictionary.
    """
    filtered_areas = {}

    if not cut_area:
        raise ValueError("Please specify the cut area.")
    
    for area_key, area_value in total_areas.items():
        filtered_area = {
            "left_lanes_area": {},
            "right_lanes_area": {},
            "types": area_value["types"]
        }
        
        # Process left lanes
        # for cut_area in cut_areas:
        x_min = cut_area[0]
        x_max = cut_area[1]
        y_min = cut_area[2]
        y_max = cut_area[3]
        for left_lane_id, left_lane_area in area_value["left_lanes_area"].items():

            # Filter inner points
            filtered_inner = [
                point for point in left_lane_area["inner"]
                # if (not (x_min < point[0] < x_max)) and (not(y_min < point[1] < y_max)) # and z_min < point[2] < z_max
                if not (x_min < point[0] < x_max and y_min < point[1] < y_max)
            ]
            # Filter outer points
            filtered_outer = [
                point for point in left_lane_area["outer"]
                # if (not (x_min < point[0] < x_max)) and (not (y_min < point[1] < y_max)) # and z_min < point[2] < z_max
                if not (x_min < point[0] < x_max and y_min < point[1] < y_max)
            ]
            # Update the filtered area
            filtered_area["left_lanes_area"][left_lane_id] = {
                "inner": filtered_inner,
                "outer": filtered_outer
            }

        # Process right lanes
        for right_lane_id, right_lane_area in area_value["right_lanes_area"].items():

            # Filter inner points
            filtered_inner = [
                point for point in right_lane_area["inner"]
                # if not (x_min < point[0] < x_max) and not (y_min < point[1] < y_max) # and z_min < point[2] < z_max
                if not (x_min < point[0] < x_max and y_min < point[1] < y_max)
            ]
            # Filter outer points
            filtered_outer = [
                point for point in right_lane_area["outer"]
                # if not (x_min < point[0] < x_max) and not (y_min < point[1] < y_max) # and z_min < point[2] < z_max
                if not (x_min < point[0] < x_max and y_min < point[1] < y_max)
            ]
            # Update the filtered area
            filtered_area["right_lanes_area"][right_lane_id] = {
                "inner": filtered_inner,
                "outer": filtered_outer
            }
    
        filtered_areas[area_key] = filtered_area
    
    return filtered_areas

def plot_planes_of_roads_filter(total_areas, save_folder, 
                                center_pose, range_x, range_y,
                                intersection_name,file):
    """
    Plot the roads, filter the coordinates, and save the filtered data as a pickle file.
    
    :param total_areas: Dictionary containing road area data.
    :param save_folder: Path to the folder where outputs will be saved.
    :return: None
    """
    # Configuration flags for plotting different elements
    plot_control_center_n_ref_line = False
    plot_boundary_line = True
    plot_lane_area = True
    save_pkl = True
    
    # Clear any existing plots
    # plt.cla()
    
    # Initialize a new figure with a large size
    plt.figure(figsize=(160, 90))
    
    # Sampling interval for scatter plots to accelerate rendering
    area_select = 10  # select one from 10 boundary points for accelerating.
    
    all_types = set()
    
    if save_pkl:
        # judge if the save_folder exists
        print("save_folder:", save_folder)
        if not os.path.exists(save_folder):
            raise FileNotFoundError("The save_folder does not exist, please specify a valid folder.")
        # Ensure the save_folder exists; create it if it doesn't
        os.makedirs(save_folder, exist_ok=True)
        print("save_folder:", save_folder)
    
    x_min, x_max, y_min, y_max = output_intersection_range_inv_y(center_pose, range_x, range_y)
    # print("x_min:",x_min)
    # print("x_max:",x_max)
    # print("y_min:",y_min)
    # print("y_max:",y_max)

    # Filter the total_areas based on the specified coordinate ranges
    filtered_total_areas = filter_total_areas(total_areas, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)
    
    # Plot lane area if enabled
    if plot_lane_area:
        for k, v in tqdm(filtered_total_areas.items(), desc="Plotting Roads"):
            left_lanes_area = v["left_lanes_area"]
            right_lanes_area = v["right_lanes_area"]
            types = v["types"]
    
            # Plot left lanes
            for left_lane_id, left_lane_area in left_lanes_area.items():
                type_of_lane = types[left_lane_id]
                all_types.add(type_of_lane)
                lane_color = TYPE_COLOR_DICT[type_of_lane]
                inner_points = left_lane_area["inner"]
                outer_points = left_lane_area["outer"]
    
                # Combine inner and outer points for    filling
                points_of_one_road = inner_points + outer_points[::-1]
                xs = [i for i, _ in points_of_one_road]
                ys = [i for _, i in points_of_one_road]
                
                # Fill the lane area with the specified color and add a label
                plt.fill(xs, ys, color=lane_color, label=type_of_lane)
                
                # Scatter plot with reduced density
                plt.scatter(xs[::area_select], ys[::area_select], color=rescale_color(lane_color, 0.5), s=1)
    
            # Plot right lanes
            for right_lane_id, right_lane_area in right_lanes_area.items():
                type_of_lane = types[right_lane_id]
                all_types.add(type_of_lane)
                lane_color = TYPE_COLOR_DICT[type_of_lane]
                inner_points = right_lane_area["inner"]
                outer_points = right_lane_area["outer"]
    
                # Combine inner and outer points for filling
                points_of_one_road = inner_points + outer_points[::-1]
                xs = [i for i, _ in points_of_one_road]
                ys = [i for _, i in points_of_one_road]
                
                # Fill the lane area with the specified color and add a label
                plt.fill(xs, ys, color=lane_color, label=type_of_lane)
                
                # Scatter plot with reduced density
                plt.scatter(xs[::area_select], ys[::area_select], color=rescale_color(lane_color, 0.5), s=1)
    
    # Plot boundaries if enabled
    if plot_boundary_line:
        for k, v in tqdm(filtered_total_areas.items(), desc="Plotting Edges"):
            left_lanes_area = v["left_lanes_area"]
            right_lanes_area = v["right_lanes_area"]
            types = v["types"]
    
            # Plot left boundaries
            for left_lane_id, left_lane_area in left_lanes_area.items():
                type_of_lane = types[left_lane_id]
                all_types.add(type_of_lane)
                lane_color = TYPE_COLOR_DICT[type_of_lane]
                inner_points = left_lane_area["inner"]
                outer_points = left_lane_area["outer"]
                
                # Combine inner and outer points for boundary
                points_of_one_road = inner_points + outer_points[::-1]
                xs = [i for i, _ in points_of_one_road]
                ys = [i for _, i in points_of_one_road]
                
                # Scatter plot for boundaries with reduced density
                plt.scatter(xs[::area_select], ys[::area_select], color=rescale_color(lane_color, 0.5), s=1)
    
            # Plot right boundaries
            for right_lane_id, right_lane_area in right_lanes_area.items():
                type_of_lane = types[right_lane_id]
                all_types.add(type_of_lane)
                lane_color = TYPE_COLOR_DICT[type_of_lane]
                inner_points = right_lane_area["inner"]
                outer_points = right_lane_area["outer"]
                
                # Combine inner and outer points for boundary
                points_of_one_road = inner_points + outer_points[::-1]
                xs = [i for i, _ in points_of_one_road]
                ys = [i for _, i in points_of_one_road]
                
                # Scatter plot for boundaries with reduced density
                plt.scatter(xs[::area_select], ys[::area_select], color=rescale_color(lane_color, 0.5), s=1)
    
    # Plot center lane and reference line if enabled
    if plot_control_center_n_ref_line:
        saved_center_lanes = dict()
        for k, v in tqdm(filtered_total_areas.items(), desc="Plotting Reference and Center"):
            reference_points = v.get("reference_points", {})
            if not reference_points:
                continue
            position_reference_points = reference_points.get("position", [])
            position_center_lane = reference_points.get("position_center_lane", [])
    
            position_reference_points_xs = [x for x, y in position_reference_points]
            position_reference_points_ys = [y for x, y in position_reference_points]
            position_center_lane_xs = [x for x, y in position_center_lane]
            position_center_lane_ys = [y for x, y in position_center_lane]
    
            saved_center_lanes[k] = position_center_lane
            plt.scatter(position_reference_points_xs, position_reference_points_ys, color=COLOR_REFERECE_LINE, s=3)
            plt.scatter(position_center_lane_xs, position_center_lane_ys, color=COLOR_CENTER_LANE, s=2)
    
    # Create legend entries
    legend_dict = create_legend_dict(all_types)

    # Parse and plot crosswalks
    crosswalks = parse_crosswalks(file)
    filtered_crosswalks = filter_crosswalks(crosswalks, x_min, x_max, y_min, y_max)
    plot_crosswalks(filtered_crosswalks, plt.gca())

    # Save the filtered_total_areas to a pickle file
    if save_pkl:
        pickle_path = os.path.join(save_folder, f"{intersection_name}_filtered_total_areas.pkl")
        with open(pickle_path, "wb") as f:
            pickle.dump(filtered_total_areas, f)
        print(f"Filtered total areas have been saved to {pickle_path}")

    if save_pkl:
        pickle_path2 = os.path.join(save_folder, f"{intersection_name}_filtered_crosswalks.pkl")
        with open(pickle_path2, "wb") as f:
            pickle.dump(filtered_crosswalks, f)
        print(f"Filtered crosswalks have been saved to {pickle_path2}")
    
    # Add legend to the plot
    plt.legend(handles=legend_dict.values(), labels=legend_dict.keys(), fontsize=8, loc='upper right')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.show()

def create_legend_dict(all_types):
    """
    Create a dictionary of legend entries for the plot.

    Parameters:
    - all_types: Set of all types of lanes in the plot.

    Returns:
    - Dictionary of legend entries for the plot.
    """
    legend_dict = {
        k: Patch(facecolor=v, edgecolor=v, alpha=1.0) for k, v in TYPE_COLOR_DICT.items() if k in all_types
    }
    legend_dict.update({
        "center_lane": Patch(facecolor=COLOR_CENTER_LANE, edgecolor=COLOR_CENTER_LANE, alpha=1.0),
    })
    legend_dict.update({
        "reference_line": Patch(facecolor=COLOR_REFERECE_LINE, edgecolor=COLOR_REFERECE_LINE, alpha=1.0),
    })
    # Add crosswalks to legend_dict 
    legend_dict.update({
        "crosswalk": Patch(facecolor=TYPE_COLOR_DICT['crosswalk'], edgecolor=TYPE_COLOR_DICT['crosswalk'], alpha=1.0),
    })    
    return legend_dict

def plot_planes_of_roads_from_pkl_filter(total_areas, crosswalks, center_pose, range_x, range_y, filter_cut = False):
    # Configuration flags for plotting different elements
    plot_control_center_n_ref_line = False
    plot_boundary_line = True
    plot_lane_area = True
    save_pkl = True

    x_min, x_max, y_min, y_max = output_intersection_range(center_pose, range_x, range_y)
    filtered_total_areas = filter_total_areas(total_areas, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)

    # ! Hardcode for tintersectino_town05
    # filter_cut = True
    
    if filter_cut:
        # ! This area is hardcode, need to be changed based on your needs.
        cut_areas = CUT_AREAS 

        # cut_areas: [[x_min,x_max,y_min,y_max],[x_min,x_max,y_min,y_max]].
        filtered_total_areas = filter_total_areas_cut_areas(filtered_total_areas, cut_areas)

    filtered_crosswalks = filter_crosswalks(crosswalks, x_min, x_max, y_min, y_max)
    # Clear any existing plots
    # plt.cla()
    
    # Initialize a new figure with a large size
    plt.figure(figsize=(16, 9))
    
    # Sampling interval for scatter plots to accelerate rendering
    area_select = 10  # select one from 10 boundary points for accelerating.
    
    all_types = set()

    # Plot lane area if enabled
    if plot_lane_area:
        for k, v in tqdm(filtered_total_areas.items(), desc="Plotting Roads"):
            left_lanes_area = v["left_lanes_area"]
            right_lanes_area = v["right_lanes_area"]
            types = v["types"]
    
            # Plot left lanes
            for left_lane_id, left_lane_area in left_lanes_area.items():
                type_of_lane = types[left_lane_id]
                all_types.add(type_of_lane)
                lane_color = TYPE_COLOR_DICT[type_of_lane]
                inner_points = left_lane_area["inner"]
                outer_points = left_lane_area["outer"]
    
                # Combine inner and outer points for    filling
                points_of_one_road = inner_points + outer_points[::-1]
                xs = [i for i, _ in points_of_one_road]
                ys = [i for _, i in points_of_one_road]
                
                # Fill the lane area with the specified color and add a label
                plt.fill(xs, ys, color=lane_color, label=type_of_lane, zorder=1)
                
                # Scatter plot with reduced density
                plt.scatter(xs[::area_select], ys[::area_select], color=rescale_color(lane_color, 0.5), s=1, zorder=2)
    
            # Plot right lanes
            for right_lane_id, right_lane_area in right_lanes_area.items():
                type_of_lane = types[right_lane_id]
                all_types.add(type_of_lane)
                lane_color = TYPE_COLOR_DICT[type_of_lane]
                inner_points = right_lane_area["inner"]
                outer_points = right_lane_area["outer"]
    
                # Combine inner and outer points for filling
                points_of_one_road = inner_points + outer_points[::-1]
                xs = [i for i, _ in points_of_one_road]
                ys = [i for _, i in points_of_one_road]
                
                # Fill the lane area with the specified color and add a label
                plt.fill(xs, ys, color=lane_color, label=type_of_lane, zorder=1)
                
                # Scatter plot with reduced density
                plt.scatter(xs[::area_select], ys[::area_select], color=rescale_color(lane_color, 0.5), s=1, zorder=2)
    
    # Plot boundaries if enabled
    if plot_boundary_line:
        for k, v in tqdm(filtered_total_areas.items(), desc="Plotting Edges"):
            left_lanes_area = v["left_lanes_area"]
            right_lanes_area = v["right_lanes_area"]
            types = v["types"]
    
            # Plot left boundaries
            for left_lane_id, left_lane_area in left_lanes_area.items():
                type_of_lane = types[left_lane_id]
                all_types.add(type_of_lane)
                lane_color = TYPE_COLOR_DICT[type_of_lane]
                inner_points = left_lane_area["inner"]
                outer_points = left_lane_area["outer"]
                
                # Combine inner and outer points for boundary
                points_of_one_road = inner_points + outer_points[::-1]
                xs = [i for i, _ in points_of_one_road]
                ys = [i for _, i in points_of_one_road]
                
                # Scatter plot for boundaries with reduced density
                plt.scatter(xs[::area_select], ys[::area_select], color=rescale_color(lane_color, 0.5), s=1)
    
            # Plot right boundaries
            for right_lane_id, right_lane_area in right_lanes_area.items():
                type_of_lane = types[right_lane_id]
                all_types.add(type_of_lane)
                lane_color = TYPE_COLOR_DICT[type_of_lane]
                inner_points = right_lane_area["inner"]
                outer_points = right_lane_area["outer"]
                
                # Combine inner and outer points for boundary
                points_of_one_road = inner_points + outer_points[::-1]
                xs = [i for i, _ in points_of_one_road]
                ys = [i for _, i in points_of_one_road]
                
                # Scatter plot for boundaries with reduced density
                plt.scatter(xs[::area_select], ys[::area_select], color=rescale_color(lane_color, 0.5), s=1)
    
    # Plot center lane and reference line if enabled
    if plot_control_center_n_ref_line:
        saved_center_lanes = dict()
        for k, v in tqdm(filtered_total_areas.items(), desc="Plotting Reference and Center"):
            reference_points = v.get("reference_points", {})
            if not reference_points:
                continue
            position_reference_points = reference_points.get("position", [])
            position_center_lane = reference_points.get("position_center_lane", [])
    
            position_reference_points_xs = [x for x, y in position_reference_points]
            position_reference_points_ys = [y for x, y in position_reference_points]
            position_center_lane_xs = [x for x, y in position_center_lane]
            position_center_lane_ys = [y for x, y in position_center_lane]
    
            saved_center_lanes[k] = position_center_lane
            plt.scatter(position_reference_points_xs, position_reference_points_ys, color=COLOR_REFERECE_LINE, s=3)
            plt.scatter(position_center_lane_xs, position_center_lane_ys, color=COLOR_CENTER_LANE, s=2)

    # Plot crosswalk
    plot_crosswalks(filtered_crosswalks, plt.gca())
    
    # create legend dict
    legend_dict = create_legend_dict(all_types)

    # set limit
    # center_pose = [-46.734985,21.328648] # x=-46.734985, y=21.328648
    offset = 10
    range_x_new = range_x + offset  
    range_y_new = range_y + offset 
    x_min, x_max, y_min, y_max = output_intersection_range_inv_y(center_pose, range_x_new, range_y_new)

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    # Add legend to the plot
    plt.legend(handles=legend_dict.values(), labels=legend_dict.keys(), fontsize=8, loc='upper right')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.show()

def plot_crosswalks(crosswalks, ax):
    """
    Draw crosswalks on the matplotlib figure.

    Parameters:
    - crosswalks: List of crosswalk objects with their outlines and properties.
    - ax: Matplotlib axis to draw on.
    """
    # Draw the crosswalks
    for crosswalk in crosswalks:
        corners = crosswalk['outline']
        xs = [corner['X'] for corner in corners]
        ys = [corner['Y'] for corner in corners]

        # Close the polygon if not closed
        if xs[0] != xs[-1] or ys[0] != ys[-1]:
            xs.append(xs[0])
            ys.append(ys[0])

        # Plot the crosswalk and add a label for the legend
        ax.fill(xs, ys, color=TYPE_COLOR_DICT['crosswalk'], label='Crosswalk', alpha=0.6)

def load_pkl_files_n_plot(total_area_file, crosswalk_file,center_pose, range_x, range_y):
    """
    Load the pickle files containing total areas and crosswalks, and plot the roads with crosswalks.

    Parameters:
    - total_area_file: Path to the pickle file containing total areas.
    - crosswalk_file: Path to the pickle file containing crosswalks.
    - save_folder: Path to the folder where outputs will be saved.
    """
    # Load the total areas and crosswalks from the pickle files
    with open(total_area_file, 'rb') as f:
        total_areas = pickle.load(f)
    with open(crosswalk_file, 'rb') as f:
        crosswalks = pickle.load(f)

    filter_flag = True
    if filter_flag:
        # Plot the roads with crosswalks
        plot_planes_of_roads_from_pkl_filter(total_areas, crosswalks,center_pose, range_x, range_y, filter_cut = False)
    else:
        # Plot the roads with crosswalks
        plot_planes_of_roads(total_areas)

def process_one_file(file,
                     center_pose, 
                     range_x, range_y,
                     step = 0.1,
                     filter_flag = False,
                     intersection_name="Town10HD_Opt"):
    """
    Load one .xodr file and calculate the railing positions with other important messages.
    :param file: Input file.
    :param step: Step of calculation.
    :return: None
    """

    assert os.path.exists(file), FileNotFoundError(file)

    # derive town name & center coords
    town_name = os.path.splitext(os.path.basename(file))[0]  # e.g. "Town10HD_Opt"
    cx, cy = center_pose

    # prepare output folder
    out_dir = os.path.join('/home/handsomeyun/BEVFormer/data/map', town_name, str(int(cx)) + '_' + str(int(cy)))
    os.makedirs(out_dir, exist_ok=True)

    print("new_save_path:", out_dir)

    road_network = load_xodr_and_parse(file)
    total_areas = get_all_lanes(road_network, step=step)
    
    if filter_flag:
        plot_planes_of_roads_filter(total_areas, out_dir,
                                    center_pose, range_x, range_y,
                                    intersection_name, file)
    else:
        plot_planes_of_roads(total_areas,out_dir,save_pkl=True)

    # plot_planes_of_roads(total_areas, save_folder)

def filter_crosswalks(crosswalks, x_min, x_max, y_min, y_max):
    """
    Filter crosswalks that are within the specified bounding box.

    Parameters:
    - crosswalks: A list of crosswalk objects containing their outlines and properties.
    - x_min, x_max, y_min, y_max: The minimum and maximum X and Y coordinates of the bounding box.

    Returns:
    - A list of crosswalks that are within the bounding box.
    """
    filtered = []
    for crosswalk in crosswalks:
        # Check if all points of the crosswalk are within the bounding box
        # corners = crosswalk['outline']
        inside = all(
            x_min <= point['X'] <= x_max and y_min <= point['Y'] <= y_max
            for point in crosswalk['outline']
        )
        if inside:
            filtered.append(crosswalk)
    return filtered

def compute_filtered_pickles(xodr_file, center_pose, range_x, range_y,
                             town_name, out_dir):
    """
    Exactly what process_one_file does internally, but RETURNS the two objects
    rather than only dumping pickles.
    """
    # ... replicate the parts of process_one_file up to the plotting step,
    # but instead of just saving, return total_areas and crosswalks:
    road_network = load_xodr_and_parse(xodr_file)
    total_areas = get_all_lanes(road_network, step=0.1)
    filtered_total_areas = filter_total_areas(total_areas,
                                              *output_intersection_range_inv_y(center_pose, range_x, range_y))
    crosswalks = parse_crosswalks(xodr_file)
    # filtered_crosswalks = filter_crosswalks(crosswalks, *output_intersection_range_inv_y(center_pose, range_x, range_y))
    # return filtered_total_areas, filtered_crosswalks
    return total_areas, crosswalks


def test2():
    fig, ax = plt.subplots(figsize=(16, 9))
    crosswalks = parse_crosswalks(XODR_FILE)

    # Plot crosswalks
    plot_crosswalks(crosswalks, ax)

    # Save the final visualization
    plt.legend()
    plt.axis('equal')
    plt.show()

# ==================================================================================================    
def main():
    # Write an argument parser to get the file path
    # parser = argparse.ArgumentParser()
    # parser.add_argument("file", help="Path to the input .xodr file")
    # parser.add_argument("--filter", action="store_true", help="Filter the coordinates")
    # args = parser.parse_args()
    # xodr_file = args.file
    # filter_flag = args.filter
    center_pose = [-62.349998474121094, 20.189998626708984] # x=-46.734985, y=21.328648
    range_x = 51.2
    range_y = 51.2
    process_one_file(file=XODR_FILE,
                     center_pose= center_pose, 
                     range_x=range_x, 
                     range_y=range_y,
                     step = 0.1,
                     filter_flag = True,
                     intersection_name="Town10HD_Opt")

def main2():
    # # This main2 is just for testing
    # total_area_file = "/home/handsomeyun/BEVFormer/data/map/Town10HD_Opt/total_areas.pkl"
    # crosswalk_file = "/home/handsomeyun/BEVFormer/data/map/Town10HD_Opt/crosswalks.pkl"
    # center_pose = [-62.349998474121094, 20.189998626708984] # x=-46.734985, y=21.328648
    # range_x = 51.2
    # range_y = 51.2
    load_pkl_files_n_plot(total_area_file, crosswalk_file, center_pose, range_x, range_y)

if __name__ == "__main__":
    # This is to generate the total_areas.pkl and crosswalks.pkl files
    # or to generate the filtered_total_areas.pkl and filtered_crosswalks.pkl files
    main() 
    # This is to load the filtered_total_areas.pkl and filtered_crosswalks.pkl files
    # and verify if the code is working correctly
    # main2()