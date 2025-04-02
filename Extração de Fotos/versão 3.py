    # -*- coding: utf-8 -*-
import os
import osmnx as ox
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString, MultiLineString, GeometryCollection
from shapely.ops import nearest_points, split, unary_union
import pandas as pd
import math
import requests
import geopandas as gpd
import numpy as np
from geopy.distance import geodesic


point_coordinates = (38.7218518, -9.4415605) #este ponto está top
# point_coordinates = (41.1669968,-8.6417347)
# point_coordinates = (41.165573, -8.645403)

# lat = float(input("Insira a latitude (separação com ponto): "))
# lon = float(input("Insira a longitude (separação com ponto): "))
folder_name = input("Insira o nome da pasta para guardar os ficheiros: ")

if not os.path.exists(folder_name):
    os.makedirs(folder_name)

# point_coordinates = (lat, lon)
dist=300

def deg_to_rad(degrees):
    return degrees * (math.pi / 180)

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    lat1 = deg_to_rad(lat1)
    lon1 = deg_to_rad(lon1)
    lat2 = deg_to_rad(lat2)
    lon2 = deg_to_rad(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

bridge_gdf = ox.features.features_from_point(point_coordinates, tags={'bridge': True, 'highway': '*'}, dist=dist)

bridge_midpoints = []
start_p_points = []
end_p_points = []
bridge_lengths = []

for geom in bridge_gdf['geometry']:
    if geom.geom_type == 'LineString':
        start_p_point = (geom.coords[0][1], geom.coords[0][0])  # Latitude e longitude do ponto de início
        end_p_point = (geom.coords[-1][1], geom.coords[-1][0])  # Latitude e longitude do ponto de fim
        bridge_midpoints.append(geom.centroid)
        start_p_points.append(start_p_point)  # Adiciona o ponto de início à lista
        end_p_points.append(end_p_point)
        print(f"Ponto de início da ponte: {start_p_point}")
        print(f"Ponto de fim da ponte: {end_p_point}")

    elif geom.geom_type == 'MultiLineString':
        for line in geom:
            start_p_point = (line.coords[0][1], line.coords[0][0])  # Latitude e longitude do ponto de início
            end_p_point = (line.coords[-1][1], line.coords[-1][0])  # Latitude e longitude do ponto de fim
            bridge_midpoints.append(line.centroid)
            start_p_points.append(start_p_point)  # Adiciona o ponto de início à lista
            end_p_points.append(end_p_point)
            print(f"Ponto de início da ponte: {start_p_point}")
            print(f"Ponto de fim da ponte: {end_p_point}")

    elif geom.geom_type == 'MultiPoint':
        for point in geom:
            bridge_midpoints.append(point)

for start, end in zip(start_p_points, end_p_points):
    # Calcular a distância entre o ponto de início e o ponto de fim
    bridge_length = haversine(start[0], start[1], end[0], end[1])
    bridge_lengths.append(bridge_length)

bridge_length = min(bridge_lengths)
print(f"A menor distância entre os pontos de início e fim é: {bridge_length:.2f} m")

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
bridge_gdf.plot(ax=ax, color='blue', alpha=0.5, label='Bridge Geometries')
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
    print("Road under Bridge:",under_bridge_segment)

circle_center = (closest_midpoint.x, closest_midpoint.y)
radius = 200
radius_degrees = radius / 111111
circle = Point(circle_center).buffer(radius_degrees)

circle_patch = plt.Circle(circle_center, radius_degrees, color='cyan', fill=False, linestyle='--', label='Circle')
ax.add_patch(circle_patch)

intersection_points = circle.intersection(under_bridge_segment)
if isinstance(intersection_points, LineString):
    intersection_coords = list(intersection_points.coords)
    start_point = intersection_coords[0]
    end_point = intersection_coords[-1]

    print("Start Point:", start_point[1],",",start_point[0])
    print("End Point:", end_point[1],",",end_point[0])
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
plt.draw()

grafico_path = os.path.join(folder_name, "OpenStreetMap.png")
plt.savefig(grafico_path, dpi=300)
plt.close(fig)

print("Closest Midpoint Coordinates:", closest_midpoint.y,",",closest_midpoint.x)


point_midpoint = (closest_midpoint.y, closest_midpoint.x)
point_start = (start_point[0], start_point[1])
point_end = (end_point[0], end_point[1])

distance_mid_start = haversine(closest_midpoint.y, closest_midpoint.x, start_point[1], start_point[0])
distance_mid_end = haversine(closest_midpoint.y, closest_midpoint.x, end_point[1], end_point[0])

print(f"Distância do midpoint até o start: {distance_mid_start:.2f} metros")
print(f"Distância do midpoint até o end: {distance_mid_end:.2f} metros")

def move_point_towards_or_away_from_midpoint(point, midpoint, distance_adjustment_factor=0.05):
    delta_lat = midpoint[0] - point[0]
    delta_lon = midpoint[1] - point[1]

    distance_h = haversine(point[0], point[1], midpoint[0], midpoint[1])
    max_move_distance = 100

    if distance_h < distance_adjustment_factor * bridge_length:
        move_distance = min(max_move_distance, distance_h * 0.1)
        point = (point[0] - delta_lat * move_distance / distance_h, point[1] - delta_lon * move_distance / distance_h)
    elif distance_h > (1 - distance_adjustment_factor) * bridge_length:
        move_distance = min(max_move_distance, distance_h * 0.1)
        point = (point[0] + delta_lat * move_distance / distance_h, point[1] + delta_lon * move_distance / distance_h)

    return point

def check_point_position(point, midpoint, bridge_length):
    distance_h = haversine(point[0], point[1], midpoint[0], midpoint[1])
    return bridge_length * 0.1 <= distance_h <= bridge_length * 0.9

#while not check_point_position(point_start, point_midpoint, bridge_length):
 #   print(f"O ponto start está fora da faixa ideal. Ajustando a posição...")
  #  point_start = move_point_towards_or_away_from_midpoint(point_start, point_midpoint, distance_adjustment_factor=0.1)
   # distance_mid_start = haversine(closest_midpoint.y, closest_midpoint.x, start_point[1], start_point[0])
    #print(f"Novo ponto start: {point_start}")

# while not check_point_position(point_end, point_midpoint, bridge_length):
   # print(f"O ponto end está fora da faixa ideal. Ajustando a posição...")
    #point_end = move_point_towards_or_away_from_midpoint(point_end, point_midpoint, distance_adjustment_factor=0.1)
  #  distance_mid_end = haversine(closest_midpoint.y, closest_midpoint.x, end_point[1], end_point[0])
   # print(f"Novo ponto end: {point_end}")

print("Ambos os pontos estão agora na distância ideal em relação à ponte!")

rua_por_baixo = "Sim" if min_distance < 20 else "Não"

info_path = os.path.join(folder_name, "Info.txt")
with open(info_path, "w", encoding="utf-8") as file:
    file.write(f"Coordenadas introduzidas: {point_coordinates}\n")
    file.write(f"Tem rua por baixo da ponte? {rua_por_baixo}\n")
    file.write(f"Ponto Start: {start_point[1],start_point[0]}\n")
    file.write(f"Ponto End: {end_point[1],end_point[0]}\n")
    file.write(f"Distância do ponto start ao ponto médio: {distance_mid_start:.2f} metros\n")
    file.write(f"Distância do ponto end ao ponto médio: {distance_mid_end: .2f} metros\n")
    file.write(f"Comprimento da ponte: {bridge_length:.2f} metros\n")
    file.write(f"Closest Midpoint Coordinates: {closest_midpoint.y,closest_midpoint.x}\n")
    file.write("\n\n")
    file.write("\n!Esta informação pode conter alguns erros!\n")

print(f"Todos os ficheiros foram guardados na pasta: {folder_name}")

#%%
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
api_key = "XXXXXXXXXXXXXXXXXXXXXXX"

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
    heading = (initial_heading + 360) % 360
    return heading

yellow_point_coordinates = (closest_midpoint.y, closest_midpoint.x)

start_lat, start_lon = start_point[1], start_point[0]
end_lat, end_lon = end_point[1], end_point[0]
midpoint_lat, midpoint_lon = closest_midpoint.y, closest_midpoint.x

delta_x_start = start_point[0] - yellow_point_coordinates[0]
delta_y_start = start_point[1] - yellow_point_coordinates[1]
heading_start = calculate_heading(start_lat, start_lon, midpoint_lat, midpoint_lon)

delta_x_end = yellow_point_coordinates[0] - end_point[0]
delta_y_end = yellow_point_coordinates[1] - end_point[1]
heading_end = calculate_heading(end_lat, end_lon, midpoint_lat, midpoint_lon)

delta_x_midpoint = closest_midpoint.x - yellow_point_coordinates[0]
delta_y_midpoint = closest_midpoint.y - yellow_point_coordinates[1]
heading_midpoint = (math.degrees(math.atan2(delta_y_midpoint, delta_x_midpoint))) % 360

file_path_start = os.path.join(folder_name,f"{folder_name}_1.png")
file_path_end = os.path.join(folder_name,f"{folder_name}_2.png")

download_street_view_image((start_point[1], start_point[0]), heading_start, pitch, fov, api_key,file_path_start)
download_street_view_image((end_point[1], end_point[0]), heading_end, pitch, fov, api_key, file_path_end)