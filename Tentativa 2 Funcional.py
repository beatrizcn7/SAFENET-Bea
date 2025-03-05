# -*- coding: utf-8 -*-
import osmnx as ox
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString, MultiLineString, GeometryCollection
from shapely.ops import nearest_points, split, unary_union
import pandas as pd
import math
import requests
import numpy as np

# point_coordinates = (38.7218518, -9.4415605)
# point_coordinates = (41.1669968,-8.6417347)
point_coordinates = (41.165606, -8.645472)
dist=500

bridge_gdf = ox.features.features_from_point(point_coordinates, tags={'bridge': True, 'highway': '*'}, dist=dist)

bridge_midpoints = []

for geom in bridge_gdf['geometry']:
    if geom.geom_type == 'LineString':
        bridge_midpoints.append(geom.centroid)
    elif geom.geom_type == 'MultiLineString':
        for line in geom:
            bridge_midpoints.append(line.centroid)
    elif geom.geom_type == 'MultiPoint':
        for point in geom:
            bridge_midpoints.append(point)

distances = [point.distance(Point(point_coordinates[1], point_coordinates[0])) for point in bridge_midpoints]

distances_df = pd.DataFrame({'Distance to Green Point (m)': distances})

closest_index = distances.index(min(distances))
closest_midpoint = bridge_midpoints[closest_index]

G = ox.graph_from_point(point_coordinates, dist=dist, network_type='drive')

edges_gdf = ox.graph_to_gdfs(G, nodes=False)

closest_edge = None
min_distance = float('inf')
for u, v, k, data in G.edges(keys=True, data=True):
    geometry = data.get('geometry')
    if geometry is None:
        edge_line = LineString([(G.nodes[u]['x'], G.nodes[u]['y']), (G.nodes[v]['x'], G.nodes[v]['y'])])
    else:
        edge_line = geometry

    closest_point_on_edge = nearest_points(edge_line, closest_midpoint)[0]
    distance_to_edge = closest_point_on_edge.distance(closest_midpoint)

    if distance_to_edge < min_distance:
        min_distance = distance_to_edge
        closest_edge = (u, v, k)

intersections = []
closest_edge_line = LineString([(G.nodes[closest_edge[0]]['x'], G.nodes[closest_edge[0]]['y']),
                                (G.nodes[closest_edge[1]]['x'], G.nodes[closest_edge[1]]['y'])])
for idx, row in edges_gdf.iterrows():
    edge_line = row['geometry']
    if edge_line.intersects(closest_edge_line):
        intersections.append(edge_line.intersection(closest_edge_line))

fig, ax = ox.plot_graph(G, edge_color='grey', edge_linewidth=0.5, show=False, node_size=0)

ax.plot(point_coordinates[1], point_coordinates[0], 'go', markersize=5, label='Coordenadas Inseridas')

for midpoint in bridge_midpoints:
    ax.plot(midpoint.x, midpoint.y, 'bo', markersize=5)

ax.plot(closest_midpoint.x, closest_midpoint.y, 'yo', markersize=5,label='Closest Midpoint')

for intersection in intersections:
    if isinstance(intersection, LineString):
        x, y = intersection.xy
        ax.plot(x, y, color='orange', linewidth=5, label='Intersections')

start_point = (G.nodes[closest_edge[0]]['x'], G.nodes[closest_edge[0]]['y'])
end_point = (G.nodes[closest_edge[1]]['x'], G.nodes[closest_edge[1]]['y'])

under_bridge_segment = None
min_distance_to_bridge = float('inf')
intersection_point = None

for idx, row in edges_gdf.iterrows():
    edge_line = row['geometry']
    if edge_line.intersects(closest_edge_line):
        intersections = edge_line.intersection(closest_edge_line)
        if isinstance(intersections, Point):
            distance_to_bridge = intersections.distance(closest_midpoint)
            if distance_to_bridge < min_distance_to_bridge:
                min_distance_to_bridge = distance_to_bridge
                under_bridge_segment = edge_line
                intersection_point = intersections

if under_bridge_segment:
    x_coords, y_coords = under_bridge_segment.xy
    ax.plot(x_coords, y_coords, color='green', linewidth=2, label='Road under Bridge')
    print("Road under Bridge:")
    print(under_bridge_segment)

circle_center = (closest_midpoint.x, closest_midpoint.y)

radius = 300
radius_degrees = radius / 111111

circle = Point(circle_center).buffer(radius_degrees)

circle_patch = plt.Circle(circle_center, radius_degrees, color='cyan', fill=False, linestyle='--', label='Circle')
ax.add_patch(circle_patch)

intersection_points = circle.intersection(under_bridge_segment)

if isinstance(intersection_points, LineString):
    intersection_coords = list(intersection_points.coords)
    start_point = intersection_coords[0]
    end_point = intersection_coords[-1]

    print("Start Point:", start_point)
    print("End Point:", end_point)
    # Faz os pontos de início e fim
    ax.scatter(start_point[0], start_point[1], color='orange', label='Start Point', s=10, zorder=5)
    ax.scatter(end_point[0], end_point[1], color='cyan', label='End Point', s=10, zorder=5)

line_segment = LineString([start_point, end_point])

x_coords, y_coords = line_segment.xy

def find_intersection_points(center_x, center_y, radius, x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    a = dx ** 2 + dy ** 2
    b = 2 * (dx * (x1 - center_x) + dy * (y1 - center_y))
    c = (x1 - center_x) ** 2 + (y1 - center_y) ** 2 - radius ** 2
    discriminant = b ** 2 - 4 * a * c

    if discriminant < 0:
        return None

    t1 = (-b + math.sqrt(discriminant)) / (2 * a)
    t2 = (-b - math.sqrt(discriminant)) / (2 * a)

    start_point_projected = (x1 + t1 * dx, y1 + t1 * dy)
    end_point_projected = (x1 + t2 * dx, y1 + t2 * dy)

    return start_point_projected, end_point_projected

def plot_intersection_points(ax, center_x, center_y, radius, x1, y1, x2, y2, intersection_points):
    if intersection_points:
        ax.plot(intersection_points[0][0], intersection_points[0][1], 'ro', label='Start point projected')
        ax.plot(intersection_points[1][0], intersection_points[1][1], 'ro', label='End point projected')

intersection_points = find_intersection_points(circle_center[0], circle_center[1], radius_degrees, start_point[0],
                                               start_point[1], end_point[0], end_point[1])

plot_intersection_points(ax, circle_center[0], circle_center[1], radius_degrees, start_point[0], start_point[1],
                         end_point[0], end_point[1], intersection_points)
start_point_projected = intersection_points[0]
end_point_projected = intersection_points[1]

ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_title('Streets')

ax.set_aspect('equal', adjustable='datalim')
ax.legend()
plt.grid(True)
plt.show()

def download_street_view_image(location, heading, pitch, fov, api_key, file_path):
    """
    Download a Street View image given the location and parameters.
    :param location: Tuple containing latitude and longitude coordinates, e.g., (latitude, longitude)
    :param heading: Compass heading indicating the orientation of the camera.
    :param pitch: The up or down angle of the camera.
    :param fov: The field of view of the camera.
    :param api_key: Your Google Cloud Platform API key.
    :param file_path: File path where the downloaded image will be saved.
    """

    base_url = "https://maps.googleapis.com/maps/api/streetview"
    parameters = {
        "size": "600x400",
        "location": f"{location[0]},{location[1]}",
        "heading": heading,
        "pitch": pitch,
        "fov": fov,
        "key": api_key
    }

    response = requests.get(base_url, params=parameters)
    if response.status_code == 200:
        with open(file_path, 'wb') as f:
            f.write(response.content)
        print(f"Image downloaded successfully at: {file_path}")
    else:
        print(f"Failed to download image. Status code: {response.status_code}")

pitch = 0
fov = 200
api_key = "AIzaSyCsVw_oDiVdzN42YzTljVmWisWrIyczFW8"

def calculate_heading(lat1, lon1, lat2, lon2):
    """
    Calculate the heading (azimuth) between two geographic points.
    The result is in degrees from the north (0 degrees).
    """
    delta_lon = lon2 - lon1
    x = math.sin(math.radians(delta_lon)) * math.cos(math.radians(lat2))
    y = math.cos(math.radians(lat1)) * math.sin(math.radians(lat2)) - \
        math.sin(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.cos(math.radians(delta_lon))

    initial_heading = math.atan2(x, y)
    initial_heading = math.degrees(initial_heading)
    heading = (initial_heading + 360) % 360  # Normalize to [0, 360)
    return heading

yellow_point_coordinates = (closest_midpoint.y, closest_midpoint.x)

start_lat, start_lon = start_point[1], start_point[0]  # (latitude, longitude) para o ponto de início
end_lat, end_lon = end_point[1], end_point[0]  # (latitude, longitude) para o ponto de fim
midpoint_lat, midpoint_lon = closest_midpoint.y, closest_midpoint.x

# Cálculo do heading para o start (já correto)
delta_x_start = start_point[0] - yellow_point_coordinates[0]
delta_y_start = start_point[1] - yellow_point_coordinates[1]
heading_start = calculate_heading(start_lat, start_lon, midpoint_lat, midpoint_lon)

# Cálculo do heading para o end (inverter a direção)
delta_x_end = yellow_point_coordinates[0] - end_point[0]  # Inverter o sinal de delta_x
delta_y_end = yellow_point_coordinates[1] - end_point[1]  # Inverter o sinal de delta_y
heading_end = calculate_heading(end_lat, end_lon, midpoint_lat, midpoint_lon)

# Cálculo do heading para o closest midpoint
delta_x_midpoint = closest_midpoint.x - yellow_point_coordinates[0]
delta_y_midpoint = closest_midpoint.y - yellow_point_coordinates[1]
heading_midpoint = (math.degrees(math.atan2(delta_y_midpoint, delta_x_midpoint))) % 360

# Mostrar os headings ajustados
print(f"Heading ajustado para o start: {heading_start}")
print(f"Heading ajustado para o end: {heading_end}")
print(f"Heading ajustado para o closest midpoint: {heading_midpoint}")

file_path_start = "p_start 1.jpg"
file_path_end = "p_end 1.jpg"

print("Heading from start point to yellow point:", heading_start)
print("Heading from end point to yellow point:", heading_end)
print("Absolute difference between headings:", abs(heading_start - heading_end))

download_street_view_image((start_point[1], start_point[0]), heading_start, pitch, fov, api_key,
                           file_path_start)
download_street_view_image((end_point[1], end_point[0]), heading_end, pitch, fov, api_key, file_path_end)