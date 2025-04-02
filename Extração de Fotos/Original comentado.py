# Importa a biblioteca OSMnx, usada para obter e analisar dados geogr�ficos de OpenStreetMap.
import osmnx as ox
# Importa o m�dulo pyplot do Matplotlib, que � utilizado para criar gr�ficos e visualizar dados.
import matplotlib.pyplot as plt
# Importa classes da biblioteca Shapely que permitem trabalhar com geometrias espaciais, como pontos, linhas e cole��es de formas geom�tricas.
from shapely.geometry import Point, LineString, MultiLineString, GeometryCollection
# Importa fun��es da Shapely para opera��es geom�tricas avan�adas, como encontrar os pontos mais pr�ximos, dividir geometrias e unir v�rias formas geom�tricas numa s�.
from shapely.ops import nearest_points, split, unary_union
# Importa a biblioteca Pandas, usada para manipular e analisar dados em tabelas, como ficheiros CSV ou bases de dados.
import pandas as pd
# Importa a biblioteca Math, que fornece fun��es matem�ticas, como c�lculos trigonom�tricos, exponenciais e logaritmos.
import math
# Importa a biblioteca Requests, usada para fazer pedidos HTTP, como aceder a dados da internet.
import requests
# Importa a biblioteca NumPy, que permite realizar c�lculos matem�ticos avan�ados e opera��es eficientes com arrays num�ricos.
import numpy as np

# Define as coordenadas do ponto de interesse (latitude, longitude)
point_coordinates = (38.7218518, -9.4415605)

dist = 200 # Raio para pesquisa de estruturas

# Carrega as geometrias das pontes a partir do OpenStreetMap com base nas coordenadas
bridge_gdf = ox.features.features_from_point(point_coordinates, tags={'bridge': True, 'highway': '*'}, dist=dist)

# Lista para armazenar os pontos m�dios das pontes
bridge_midpoints = []

for geom in bridge_gdf['geometry']:
    if geom.geom_type == 'LineString':
        bridge_midpoints.append(geom.centroid)  # Adiciona o centro da linha
    elif geom.geom_type == 'MultiLineString':
        for line in geom:
            bridge_midpoints.append(line.centroid)  # Adiciona o centro de cada segmento
    elif geom.geom_type == 'MultiPoint':
        for point in geom:
            bridge_midpoints.append(point)  # Adiciona os pontos individuais

# Calcula a dist�ncia de cada ponto m�dio da ponte at� ao ponto de interesse
distances = [point.distance(Point(point_coordinates[1], point_coordinates[0])) for point in bridge_midpoints]

# Cria um DataFrame para armazenar as dist�ncias (opcional)
distances_df = pd.DataFrame({'Distance to Green Point (m)': distances})

# Encontra o �ndice do ponto m�dio mais pr�ximo
closest_index = distances.index(min(distances))
closest_midpoint = bridge_midpoints[closest_index]

# Obt�m o gr�fico da rede vi�ria ao redor do ponto de interesse
G = ox.graph_from_point(point_coordinates, dist=dist, network_type='drive')

# Converte as arestas do gr�fico num GeoDataFrame para opera��es geom�tricas mais f�ceis
edges_gdf = ox.graph_to_gdfs(G, nodes=False)

# Define a aresta mais pr�xima e inicializa a menor dist�ncia
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

# Lista para armazenar interse��es
intersections = []
closest_edge_line = LineString([(G.nodes[closest_edge[0]]['x'], G.nodes[closest_edge[0]]['y']),
                                (G.nodes[closest_edge[1]]['x'], G.nodes[closest_edge[1]]['y'])])
for idx, row in edges_gdf.iterrows():
    edge_line = row['geometry']
    if edge_line.intersects(closest_edge_line):
        intersections.append(edge_line.intersection(closest_edge_line))

# Faz o gr�fico da rede
fig, ax = ox.plot_graph(G, edge_color='grey', edge_linewidth=0.5, show=False, node_size=0)

# Faz o gr�fico da geometria da ponte, pontos m�dios e ponto de interesse
bridge_gdf.plot(ax=ax, color='blue', alpha=0.5, label='Bridge Geometries')
ax.plot(point_coordinates[1], point_coordinates[0], 'go', markersize=10, label='Point of Interess')

# Faz os pontos m�dios em azul
for midpoint in bridge_midpoints:
    ax.plot(midpoint.x, midpoint.y, 'bo', markersize=5)

ax.plot(closest_midpoint.x, closest_midpoint.y, 'yo', markersize=5,label='Closest Midpoint')

# Faz gr�ficos das interse��es a laranja.
for intersection in intersections:
    if isinstance(intersection, LineString):
        x, y = intersection.xy
        ax.plot(x, y, color='orange', linewidth=5, label='Intersections')

# Coordenads dos pontos finais mais pr�ximos da reta
start_point = (G.nodes[closest_edge[0]]['x'], G.nodes[closest_edge[0]]['y'])
end_point = (G.nodes[closest_edge[1]]['x'], G.nodes[closest_edge[1]]['y'])

# Inicializa vari�veis para encontrar a estrada que passa por baixo da ponte
under_bridge_segment = None
min_distance_to_bridge = float('inf')
intersection_point = None

# Encontra a estrada que passa debaixo da ponte
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

# Faz o desenho da rua que passa debaixo da ponte
if under_bridge_segment:
    x_coords, y_coords = under_bridge_segment.xy
    ax.plot(x_coords, y_coords, color='green', linewidth=2, label='Road under Bridge')

    print("Road under Bridge:")
    print(under_bridge_segment)

# Define o centro do c�rculo a partir das coordenadas do ponto m�dio mais pr�ximo
circle_center = (closest_midpoint.x, closest_midpoint.y)

# Define o raio do c�rculo em graus (assumindo coordenadas de latitude e longitude)
radius = 100
radius_degrees = radius / 111111

# Cria a geometria do c�rculo, utilizando um buffer no ponto central
circle = Point(circle_center).buffer(radius_degrees)

# Faz o c�rculo no gr�fico (com estilo e cor definidos)
circle_patch = plt.Circle(circle_center, radius_degrees, color='cyan', fill=False, linestyle='--', label='Circle')
ax.add_patch(circle_patch)

# Encontra os pontos de interse��o entre o c�rculo e o segmento de estrada abaixo da ponte
intersection_points = circle.intersection(under_bridge_segment)

# if isinstance(intersection_points, LineString):
#     for x, y in zip(x_coords, y_coords):
#         ax.scatter(x, y, color='grey', label='Road Point', s=50, zorder=5)
# elif isinstance(intersection_points, GeometryCollection):
#     for geom in intersection_points:
#         if isinstance(geom, LineString):
#             for x, y in zip(geom.xy[0], geom.xy[1]):
#                 ax.scatter(x, y, color='grey', label='Road Point', s=50, zorder=5)

# Extrai os pontos de in�cio e fim da interse��o
if isinstance(intersection_points, LineString):
    intersection_coords = list(intersection_points.coords)
    start_point = intersection_coords[0]
    end_point = intersection_coords[-1]

    print("Start Point:", start_point)
    print("End Point:", end_point)
    # Faz os pontos de in�cio e fim
    ax.scatter(start_point[0], start_point[1], color='orange', label='Start Point', s=10, zorder=5)
    ax.scatter(end_point[0], end_point[1], color='cyan', label='End Point', s=10, zorder=5)

# Cria um objeto LineString representando o segmento de linha original
line_segment = LineString([start_point, end_point])

# Extrai as coordenadas do segmento de linha e representa-as
x_coords, y_coords = line_segment.xy

# Represent a linha original do segmento
# plt.plot(x_coords, y_coords, color='red', label='Original Line Segment')

# Calcula os pontos de interse��o entre uma linha e um c�rculo, com f�rmula resolvente
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

# Fun��o para plotar os pontos de interse��o
def plot_intersection_points(ax, center_x, center_y, radius, x1, y1, x2, y2, intersection_points):
    # ax.plot([x1, x2], [y1, y2], color='red', label='Segment')
    if intersection_points:
        ax.plot(intersection_points[0][0], intersection_points[0][1], 'ro', label='Start point projected')
        ax.plot(intersection_points[1][0], intersection_points[1][1], 'ro', label='End point projected')

# fig, ax = plt.subplots()

# C�lculo dos pontos de interse��o
intersection_points = find_intersection_points(circle_center[0], circle_center[1], radius_degrees, start_point[0],
                                               start_point[1], end_point[0], end_point[1])

# Desenha os pontos no gr�fico
plot_intersection_points(ax, circle_center[0], circle_center[1], radius_degrees, start_point[0], start_point[1],
                         end_point[0], end_point[1], intersection_points)
start_point_projected = intersection_points[0]
end_point_projected = intersection_points[1]

# Configura��o do gr�fico
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_title('Streets')

# ax.autoscale()
ax.set_aspect('equal', adjustable='datalim')

ax.legend()

plt.grid(True)
plt.show()

# Download de imagens do Street View
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
api_key = "XXXXXXXXXXXXXXXXXXXXXXX"  # Replace with your own API key

yellow_point_coordinates = (closest_midpoint.y, closest_midpoint.x)

# Calcula os �ngulos de dire��o para os pontos inicial e final
delta_x_yellow_start = yellow_point_coordinates[0] - start_point_projected[0]
delta_y_yellow_start = yellow_point_coordinates[1] - start_point_projected[1]
heading_yellow_start = (math.degrees(math.atan2(delta_y_yellow_start, delta_x_yellow_start))) % 360

delta_x_yellow_end = yellow_point_coordinates[0] - end_point_projected[0]
delta_y_yellow_end = yellow_point_coordinates[1] - end_point_projected[1]
heading_yellow_end = (math.degrees(math.atan2(delta_y_yellow_end, delta_x_yellow_end))) % 360

# Caminhos das fotos
file_path_start = "street_view_image_start.jpg"
file_path_end = "street_view_image_end.jpg"

# Verifica se os �ngulos de heading s�o opostos � dire��o do ponto amarelo
print("Heading from start point to yellow point:", heading_yellow_start)
print("Heading from end point to yellow point:", heading_yellow_end)
print("Absolute difference between headings:", abs(heading_yellow_start - heading_yellow_end))

# Verifica se a diferen�a absoluta entre headings � menor que 180 graus
# if abs(heading_yellow_start - heading_yellow_end) < 180:
# print("Heading adjustment needed.")
# heading_yellow_start = (heading_yellow_start + 180) % 360
# print("Adjusted heading from start point to yellow point:", heading_yellow_start)

# Chama a fun��o para baixar as imagens do Street View
download_street_view_image((start_point_projected[1], start_point_projected[0]), heading_yellow_start, pitch,
                           fov, api_key, file_path_start)
download_street_view_image((end_point_projected[1], end_point_projected[0]), heading_yellow_end, pitch, fov,
                           api_key,file_path_end)