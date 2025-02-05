# Importa a biblioteca OSMnx, usada para obter e analisar dados geográficos de OpenStreetMap.
import osmnx as ox
# Importa o módulo pyplot do Matplotlib, que é utilizado para criar gráficos e visualizar dados.
import matplotlib.pyplot as plt
# Importa classes da biblioteca Shapely que permitem trabalhar com geometrias espaciais, como pontos, linhas e coleções de formas geométricas.
from shapely.geometry import Point, LineString, MultiLineString, GeometryCollection
# Importa funções da Shapely para operações geométricas avançadas, como encontrar os pontos mais próximos, dividir geometrias e unir várias formas geométricas numa só.
from shapely.ops import nearest_points, split, unary_union
# Importa a biblioteca Pandas, usada para manipular e analisar dados em tabelas, como ficheiros CSV ou bases de dados.
import pandas as pd
# Importa a biblioteca Math, que fornece funções matemáticas, como cálculos trigonométricos, exponenciais e logaritmos.
import math
# Importa a biblioteca Requests, usada para fazer pedidos HTTP, como aceder a dados da internet.
import requests
# Importa a biblioteca NumPy, que permite realizar cálculos matemáticos avançados e operações eficientes com arrays numéricos.
import numpy as np

# Define the coordinates for the point of interest
point_coordinates = (41.181221, -8.593758)
point_coordinates = (40.646458, -8.596817)
# Aveiro: 40.646458, -8.596817
# Porto:  41.1713908, -8.59774607

dist = 400

# Load the bridge geometries by specifying geographic coordinates (latitude and longitude)
bridge_gdf = ox.features.features_from_point(point_coordinates, tags={'bridge': True, 'highway': '*'}, dist=dist)

# Calculate midpoints of each bridge geometry
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

# Calculate distances of midpoints to the point of interest
distances = [point.distance(Point(point_coordinates[1], point_coordinates[0])) for point in bridge_midpoints]

# Create a DataFrame to store distances (optional)
distances_df = pd.DataFrame({'Distance to Green Point (m)': distances})

# Find the index of the closest midpoint
closest_index = distances.index(min(distances))
closest_midpoint = bridge_midpoints[closest_index]

# Plot the original graph
G = ox.graph_from_point(point_coordinates, dist=dist, network_type='drive')

# Convert graph edges to GeoDataFrame for easier geometric operations
edges_gdf = ox.graph_to_gdfs(G, nodes=False)

# Define the closest edge LineString
closest_edge = None
min_distance = float('inf')  # Initialize with a large value
for u, v, k, data in G.edges(keys=True, data=True):
    geometry = data.get('geometry')
    if geometry is None:
        # If the edge has no geometry, calculate the LineString from node coordinates
        edge_line = LineString([(G.nodes[u]['x'], G.nodes[u]['y']), (G.nodes[v]['x'], G.nodes[v]['y'])])
    else:
        # Otherwise, use the edge geometry
        edge_line = geometry

    # Calculate the closest point on the edge to the midpoint
    closest_point_on_edge = nearest_points(edge_line, closest_midpoint)[0]

    # Calculate the distance between the closest point on the edge and the midpoint
    distance_to_edge = closest_point_on_edge.distance(closest_midpoint)

    # Update the closest edge if this edge has a shorter distance
    if distance_to_edge < min_distance:
        min_distance = distance_to_edge
        closest_edge = (u, v, k)

# Check for intersections with other road edges
intersections = []
closest_edge_line = LineString([(G.nodes[closest_edge[0]]['x'], G.nodes[closest_edge[0]]['y']),
                                (G.nodes[closest_edge[1]]['x'], G.nodes[closest_edge[1]]['y'])])
for idx, row in edges_gdf.iterrows():
    edge_line = row['geometry']
    if edge_line.intersects(closest_edge_line):
        intersections.append(edge_line.intersection(closest_edge_line))

# Plot the original graph
fig, ax = ox.plot_graph(G, edge_color='grey', edge_linewidth=0.5, show=False, node_size=0)

# Plot the bridge geometries, midpoints, and point of interest
bridge_gdf.plot(ax=ax, color='blue', alpha=0.5, label='Bridge Geometries')  # Plot bridge geometries
ax.plot(point_coordinates[1], point_coordinates[0], 'go', markersize=10, label='Point of Interest')  # Green dot for POI

# Plot midpoints in blue, increased size for better visibility
for midpoint in bridge_midpoints:
    ax.plot(midpoint.x, midpoint.y, 'bo', markersize=5)  # Blue dot for midpoint

ax.plot(closest_midpoint.x, closest_midpoint.y, 'yo', markersize=5,
        label='Closest Midpoint')  # Yellow dot for closest midpoint

# Plot intersections with the closest edge
for intersection in intersections:
    if isinstance(intersection, LineString):
        x, y = intersection.xy
        ax.plot(x, y, color='orange', linewidth=5, label='Intersections')

# Get coordinates of the closest edge endpoints
start_point = (G.nodes[closest_edge[0]]['x'], G.nodes[closest_edge[0]]['y'])
end_point = (G.nodes[closest_edge[1]]['x'], G.nodes[closest_edge[1]]['y'])

# Find the road segment that passes under the bridge and intersects with it
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

# Plot the road segment passing under the bridge
if under_bridge_segment:
    x_coords, y_coords = under_bridge_segment.xy
    ax.plot(x_coords, y_coords, color='green', linewidth=2,
            label='Road under Bridge')  # Plotting the road segment in green

    # Print the road segment passing under the bridge
    print("Road under Bridge:")
    print(under_bridge_segment)

# Define the center of the circle
circle_center = (closest_midpoint.x, closest_midpoint.y)

# Define the radius of the circle in degrees (assuming latitude and longitude coordinates)
radius = 50
radius_degrees = radius / 111111  # Approximate conversion from meters to degrees

# Create the circle geometry
circle = Point(circle_center).buffer(radius_degrees)

# Plot the circle
circle_patch = plt.Circle(circle_center, radius_degrees, color='cyan', fill=False, linestyle='--', label='Circle')
ax.add_patch(circle_patch)

# Find the intersection points between the circle and the road segment passing under the bridge
intersection_points = circle.intersection(under_bridge_segment)

# # Plot the road points inside the circle
# if isinstance(intersection_points, LineString):
#     for x, y in zip(x_coords, y_coords):
#         ax.scatter(x, y, color='grey', label='Road Point', s=50, zorder=5)
# elif isinstance(intersection_points, GeometryCollection):
#     for geom in intersection_points:
#         if isinstance(geom, LineString):
#             for x, y in zip(geom.xy[0], geom.xy[1]):
#                 ax.scatter(x, y, color='grey', label='Road Point', s=50, zorder=5)


# Extract Start and End coordinates from the intersection segment
if isinstance(intersection_points, LineString):
    intersection_coords = list(intersection_points.coords)
    start_point = intersection_coords[0]  # First point
    end_point = intersection_coords[-1]  # Last point

    # Print the coordinates of the start and end points
    print("Start Point:", start_point)
    print("End Point:", end_point)

    # Plot the start and end points
    ax.scatter(start_point[0], start_point[1], color='orange', label='Start Point', s=10, zorder=5)
    ax.scatter(end_point[0], end_point[1], color='cyan', label='End Point', s=10, zorder=5)

# Create a LineString object representing the original line segment
line_segment = LineString([start_point, end_point])

# Extract x and y coordinates from the original line segment
x_coords, y_coords = line_segment.xy

# Plot the original line segment
plt.plot(x_coords, y_coords, color='red', label='Original Line Segment')


def find_intersection_points(center_x, center_y, radius, x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1

    # Calculate the coefficients of the quadratic equation
    a = dx ** 2 + dy ** 2
    b = 2 * (dx * (x1 - center_x) + dy * (y1 - center_y))
    c = (x1 - center_x) ** 2 + (y1 - center_y) ** 2 - radius ** 2

    discriminant = b ** 2 - 4 * a * c

    if discriminant < 0:
        # No intersection
        return None

    # Calculate intersection points
    t1 = (-b + math.sqrt(discriminant)) / (2 * a)
    t2 = (-b - math.sqrt(discriminant)) / (2 * a)

    start_point_projected = (x1 + t1 * dx, y1 + t1 * dy)
    end_point_projected = (x1 + t2 * dx, y1 + t2 * dy)

    return start_point_projected, end_point_projected


def plot_intersection_points(ax, center_x, center_y, radius, x1, y1, x2, y2, intersection_points):
    # Plot line segment
    ax.plot([x1, x2], [y1, y2], color='red', label='Segment')

    # Plot intersection points if they exist
    if intersection_points:
        ax.plot(intersection_points[0][0], intersection_points[0][1], 'ro',
                label='Start point projected')  # 'ro' for red circles
        ax.plot(intersection_points[1][0], intersection_points[1][1], 'ro',
                label='End point projected')  # 'ro' for red circles


# Create the existing plot
# fig, ax = plt.subplots()

# Call the function to plot segment and intersection points
intersection_points = find_intersection_points(circle_center[0], circle_center[1], radius_degrees, start_point[0],
                                               start_point[1], end_point[0], end_point[1])
plot_intersection_points(ax, circle_center[0], circle_center[1], radius_degrees, start_point[0], start_point[1],
                         end_point[0], end_point[1], intersection_points)

start_point_projected = intersection_points[0]
end_point_projected = intersection_points[1]

# Add labels and title
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_title('Streets')

# Set axis limits and aspect ratio
# ax.autoscale()
ax.set_aspect('equal', adjustable='datalim')

# Show the legend
ax.legend()

# Show the plot
plt.grid(True)
plt.show()


# %%

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


pitch = 0  # Pitch (in degrees)
fov = 200  # Field of view (in degrees)
api_key = "xxxxx"  # Replace with your own API key

# Define the location (yellow point coordinates)
yellow_point_coordinates = (closest_midpoint.y, closest_midpoint.x)  # Latitude and longitude coordinates

# Calculate the heading towards the yellow point from the start and end points
delta_x_yellow_start = yellow_point_coordinates[0] - start_point_projected[
    0]  # Corrected to use longitude (x-coordinate)
delta_y_yellow_start = yellow_point_coordinates[1] - start_point_projected[
    1]  # Corrected to use latitude (y-coordinate)
heading_yellow_start = (math.degrees(math.atan2(delta_y_yellow_start, delta_x_yellow_start))) % 360

delta_x_yellow_end = yellow_point_coordinates[0] - end_point_projected[0]  # Corrected to use longitude (x-coordinate)
delta_y_yellow_end = yellow_point_coordinates[1] - end_point_projected[1]  # Corrected to use latitude (y-coordinate)
heading_yellow_end = (math.degrees(math.atan2(delta_y_yellow_end, delta_x_yellow_end))) % 360

# Define file paths for the two images
file_path_start = "./street_view_image_start.jpg"
file_path_end = "./street_view_image_end.jpg"

# Check if the heading angles are opposite to the direction of the yellow point
print("Heading from start point to yellow point:", heading_yellow_start)
print("Heading from end point to yellow point:", heading_yellow_end)
print("Absolute difference between headings:", abs(heading_yellow_start - heading_yellow_end))

# Check if the absolute difference between headings is less than 180 degrees
# if abs(heading_yellow_start - heading_yellow_end) < 180:
# print("Heading adjustment needed.")
# heading_yellow_start = (heading_yellow_start + 180) % 360     # Add 180 degrees to one of the headings
# print("Adjusted heading from start point to yellow point:", heading_yellow_start)

# Call the function to download the Street View images
download_street_view_image((start_point_projected[1], start_point_projected[0]), heading_yellow_start, pitch, fov,
                           api_key, file_path_start)
download_street_view_image((end_point_projected[1], end_point_projected[0]), heading_yellow_end, pitch, fov, api_key,
                           file_path_end)
