# -*- coding: utf-8 -*-
import math

# Função para converter graus em radianos
def deg_to_rad(degrees):
    return degrees * (math.pi / 180)

# Função de Haversine para calcular a distância entre dois pontos
def haversine(lat1, lon1, lat2, lon2):
    # Raio da Terra em metros
    R = 6371000

    # Converter as coordenadas de graus para radianos
    lat1 = deg_to_rad(lat1)
    lon1 = deg_to_rad(lon1)
    lat2 = deg_to_rad(lat2)
    lon2 = deg_to_rad(lon2)

    # Diferença das coordenadas
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Fórmula de Haversine
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    # Distância final em quilômetros
    distance = R * c
    return distance

# Entrada do usuário
lat1 = float(input("Digite a latitude do ponto de início: "))
lon1 = float(input("Digite a longitude do ponto de início: "))
lat2 = float(input("Digite a latitude do ponto final: "))
lon2 = float(input("Digite a longitude do ponto final: "))

# Calcular a distância
distance = haversine(lat1, lon1, lat2, lon2)

# Exibir o resultado
print(f"A distância entre os pontos é: {distance:.2f} m")
