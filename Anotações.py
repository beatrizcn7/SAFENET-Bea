import os
import pandas as pd
import json


excel_path = 'Dados.xlsx'

diretorio_principal = 'Fotos'

diretorio_json = 'JSON'

try:
    df = pd.read_excel(excel_path, sheet_name='Resultados')
except FileNotFoundError:
    print(f"Erro: Arquivo '{excel_path}' não encontrado.")
    exit(1)
except Exception as e:
    print(f"Erro ao ler arquivo '{excel_path}': {e}")
    exit(1)

if 'Id' not in df.columns:
    print(f"Erro: Coluna 'Id' não encontrada no DataFrame.")
    exit(1)

for bridge_id_int in range(1, 2557):
    bridge_id_str = str(bridge_id_int)
    pasta_ponte = os.path.join(diretorio_principal, bridge_id_str)

    if not os.path.isdir(pasta_ponte):
        print(f"Aviso: Pasta não encontrada para ponte '{bridge_id_str}' em '{pasta_ponte}'.")
        continue

    try:
        linha_ponte = df[df['Id'] == bridge_id_int].iloc[0]
    except IndexError:
        print(f"Aviso: Identificador '{bridge_id_str}' não encontrado no DataFrame.")
        continue

    tipo_estrutura = linha_ponte['TEC - Tipo de Estrutura']
    intervalo_anos = linha_ponte['Intervalo de Anos']
    material = linha_ponte['CON - Material 1']

    ponte_info = {
        'Id': bridge_id_int,
        'Tipo de Estrutura': tipo_estrutura,
        'Intervalo de Anos': intervalo_anos,
        'Material': material
    }

    json_dir = os.path.join(diretorio_json, bridge_id_str)
    os.makedirs(json_dir, exist_ok=True)

    json_base_path = os.path.join(json_dir, f'{bridge_id_str}.json')

    with open(json_base_path, 'w', encoding='utf-8') as json_file:
        json.dump(ponte_info, json_file, indent=4, ensure_ascii=False)

    print(f"Arquivo JSON base criado para ponte '{bridge_id_str}': '{json_base_path}'.")

    fotos_ponte = [f for f in os.listdir(pasta_ponte) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))]
    for foto in fotos_ponte:
        foto_nome = os.path.splitext(foto)[0]
        json_foto_path = os.path.join(json_dir, f'{foto_nome}.json')

        with open(json_base_path, 'r', encoding='utf-8') as json_file:
            json_data = json.load(json_file)

        with open(json_foto_path, 'w', encoding='utf-8') as json_file:
            json.dump(json_data, json_file, indent=4, ensure_ascii=False)

        print(f"Arquivo JSON copiado e renomeado para foto '{foto}' da ponte '{bridge_id_str}': '{json_foto_path}'.")

print("Processo concluído!")
