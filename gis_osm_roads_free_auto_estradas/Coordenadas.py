# -*- coding: utf-8 -
# Beatriz Neves, 10 de janeiro de 2025
# Input = JSON das pontes da autoestrada
# Output = Excel Autoestrada
# Objetivo = Retirar informação de cada ponte da autoestrada

import json
import pandas as pd

# Carregar o arquivo JSON
with open("gis_osm_roads_free_auto_estradas.json", "r") as f:
    data = json.load(f)

# Criar uma lista de dados com osm_id e outros atributos
dados = []
for feature in data:
    osm_id = feature.get('osm_id')  # Pega o osm_id
    fclass = feature.get('fclass')  # Classe da estrada
    name = feature.get('name')  # Nome da estrada
    ref = feature.get('ref')  # Referência
    maxspeed = feature.get('maxspeed')  # Velocidade máxima
    bridge = feature.get('bridge')  # Se é ponte
    tunnel = feature.get('tunnel')  # Se é túnel

    dados.append({
        "osm_id": osm_id,
        "fclass": fclass,
        "name": name,
        "ref": ref,
        "maxspeed": maxspeed,
        "bridge": bridge,
        "tunnel": tunnel
    })

# Criar um DataFrame com os dados
df = pd.DataFrame(dados)

# Salvar o DataFrame em um arquivo Excel
df.to_excel("Autoestrada.xlsx", index=False)
