import os
import pandas as pd
import json

# Caminho para o ficheiro Excel e diretórios
excel_path = 'Dados.xlsx'
diretorio_principal = 'Fotos'
diretorio_json = 'JSON'

# Tenta carregar o Excel
try:
    df = pd.read_excel(excel_path, sheet_name='Resultados')
except FileNotFoundError:
    print(f"Erro: '{excel_path}' não encontrado.")
    exit(1)
except Exception as e:
    print(f"Erro ao ler '{excel_path}': {e}")
    exit(1)

# Verifica se a coluna 'Id' existe no DataFrame
if 'Id' not in df.columns:
    print(f"Erro: Coluna 'Id' não encontrada")
    exit(1)

# Loop por cada ponte
for bridge_id_int in range(1, 2557):
    bridge_id_str = str(bridge_id_int)
    pasta_ponte = os.path.join(diretorio_principal, bridge_id_str)

    # Verifica se a pasta da ponte existe
    if not os.path.isdir(pasta_ponte):
        print(f"Aviso: Pasta não encontrada para ponte '{bridge_id_str}'")
        continue

    # Tenta encontrar a linha correspondente no Excel
    try:
        linha_ponte = df[df['Id'] == bridge_id_int].iloc[0]
    except IndexError:
        print(f"Aviso: Id '{bridge_id_str}' não encontrado")
        continue

    # Informações da ponte
    tipo_estrutura = linha_ponte['TEC - Tipo de Estrutura']
    intervalo_anos = linha_ponte['Intervalo de Anos']
    material = linha_ponte['CON - Material 1']

    # Lista de fotos na pasta da ponte
    fotos_ponte = [f for f in os.listdir(pasta_ponte) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))]

    # Cria JSONs apenas para as fotos
    for foto in fotos_ponte:
        foto_nome = os.path.splitext(foto)[0]
        json_foto_path = os.path.join(diretorio_json, bridge_id_str, f'{foto_nome}.json')

        # Garante que o diretório para os JSONs existe
        os.makedirs(os.path.dirname(json_foto_path), exist_ok=True)

        # Informações a serem gravadas no JSON
        ponte_info = {
            'Id': bridge_id_int,
            'Tipo de Estrutura': tipo_estrutura,
            'Intervalo de Anos': intervalo_anos,
            'Material': material
        }

        # Escreve o arquivo JSON com as informações da ponte
        with open(json_foto_path, 'w', encoding='utf-8') as json_file:
            json.dump(ponte_info, json_file, indent=4, ensure_ascii=False)

        print(f"JSON criado para '{foto}' da ponte '{bridge_id_str}': '{json_foto_path}'.")

print("!!CONCLUÍDO!!")
