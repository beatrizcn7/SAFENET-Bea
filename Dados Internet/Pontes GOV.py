import requests
import pandas as pd
import json

# URL do JSON fornecido
url = "https://dadosabertos.cascais.pt/dataset/0796f9bd-4e99-4acd-b948-659c51ff98b3/resource/1912852d-037c-4f4d-968c-54935087e716/download/cbpassagemsuperior.json"

# Baixar o JSON
response = requests.get(url)
data = response.json()

# Lista para armazenar os dados
dados = []

# Iterar sobre cada feature
for feature in data['features']:
    # Extrair as propriedades
    properties = feature['properties']
    geometry = feature['geometry']

    # Coletar os valores necessários
    nome = properties.get("Name", "N/A")
    descricao = properties.get("description", "N/A")
    altitudemode = properties.get("altitudeMode", "N/A")
    tessellate = properties.get("tessellate", "N/A")
    extrude = properties.get("extrude", "N/A")
    visibility = properties.get("visibility", "N/A")

    # Coletar as coordenadas
    coords = geometry['coordinates'][0]  # Assumindo que sempre haverá coordenadas
    latitude = coords[0][1]  # Latitude do primeiro ponto
    longitude = coords[0][0]  # Longitude do primeiro ponto
    coordenadas = f"{latitude},{longitude}"  # Formato solicitado (latitude, longitude)

    # Adicionar à lista de dados
    dados.append({
        'Nome': nome,
        'Descrição': descricao,
        'AltitudeMode': altitudemode,
        'Tessellate': tessellate,
        'Extrude': extrude,
        'Visibility': visibility,
        'Coordenadas': coordenadas
    })

# Criar o DataFrame do Pandas
df = pd.DataFrame(dados)

# Salvar o DataFrame como um arquivo Excel
df.to_excel("Dados Internet/Pontes GOV.xlsx", index=False)

print("Criado!")
