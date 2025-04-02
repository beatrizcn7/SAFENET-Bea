# Beatriz Neves, 10 de janeiro de 2025
# Input = Excel das Coordenadas
# Output = Excel Autoestrada
# Objetivo = Coloca as cooredenadas de cada ponte no ExceL Autoestrada

import pandas as pd
import json

# Carregar o Excel com as coordenadas
excel_file = "gis_osm_roads_free_auto_estradas/Coordenadas.xlsx"
df_excel = pd.read_excel(excel_file)

# Carregar o ficheiro JSON
json_file = "gis_osm_roads_free_auto_estradas.json"
with open(json_file, "r", encoding="utf-8") as f:
    json_data = json.load(f)

# Converter os dados JSON para um DataFrame
df_json = pd.DataFrame(json_data)

# Converter as colunas 'osm_id' para string
df_excel['osm_id'] = df_excel['osm_id'].astype(str)
df_json['osm_id'] = df_json['osm_id'].astype(str)

# Verificar se o campo osm_id existe em ambos os ficheiros
if "osm_id" not in df_excel.columns or "osm_id" not in df_json.columns:
    raise ValueError("O campo 'osm_id' não está presente em ambos os ficheiros.")

# Juntar os dados com base no osm_id
df_final = pd.merge(df_excel, df_json, on="osm_id", how="inner")

# Salvar o resultado num ficheiro Excel
output_file = "Autoestrada.xlsx"
df_final.to_excel(output_file, index=False)

print(f"Dados combinados salvos no ficheiro: {output_file}")
