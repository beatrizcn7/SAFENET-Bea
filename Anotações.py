import os
import pandas as pd
import json

# Caminho para o arquivo Excel
excel_path = 'Dados.xlsx'

# Diretório principal onde estão as pastas com identificadores de pontes
diretorio_principal = 'Fotos'

# Diretório para armazenar os arquivos JSON
diretorio_json = 'JSON'

# Carregar o arquivo Excel para um DataFrame do Pandas
try:
    df = pd.read_excel(excel_path, sheet_name='Resultados')
except FileNotFoundError:
    print(f"Erro: Arquivo '{excel_path}' não encontrado.")
    exit(1)
except Exception as e:
    print(f"Erro ao ler arquivo '{excel_path}': {e}")
    exit(1)

# Verificar se 'Id' está presente nas colunas do DataFrame
if 'Id' not in df.columns:
    print(f"Erro: Coluna 'Id' não encontrada no DataFrame.")
    exit(1)

# Loop pelos identificadores de pontes (de 1 a 2556)
for bridge_id_int in range(1, 2557):
    bridge_id_str = str(bridge_id_int)
    pasta_ponte = os.path.join(diretorio_principal, bridge_id_str)

    if not os.path.isdir(pasta_ponte):
        print(f"Aviso: Pasta não encontrada para ponte '{bridge_id_str}' em '{pasta_ponte}'.")
        continue

    # Procurar a linha correspondente ao identificador da ponte no DataFrame
    try:
        linha_ponte = df[df['Id'] == bridge_id_int].iloc[0]
    except IndexError:
        print(f"Aviso: Identificador '{bridge_id_str}' não encontrado no DataFrame.")
        continue

    # Extrair informações do DataFrame
    tipo_estrutura = linha_ponte['TEC - Tipo de Estrutura']
    intervalo_anos = linha_ponte['Intervalo de Anos']
    material = linha_ponte['CON - Material 1']

    # Criar dicionário com as informações da ponte
    ponte_info = {
        'Id': bridge_id_int,
        'Tipo de Estrutura': tipo_estrutura,
        'Intervalo de Anos': intervalo_anos,
        'Material': material
    }

    # Criar diretório para os arquivos JSON da ponte se não existir
    json_dir = os.path.join(diretorio_json, bridge_id_str)
    os.makedirs(json_dir, exist_ok=True)

    # Caminho para o arquivo JSON base
    json_base_path = os.path.join(json_dir, f'{bridge_id_str}.json')

    # Escrever as informações da ponte no arquivo JSON base
    with open(json_base_path, 'w', encoding='utf-8') as json_file:
        json.dump(ponte_info, json_file, indent=4, ensure_ascii=False)

    print(f"Arquivo JSON base criado para ponte '{bridge_id_str}': '{json_base_path}'.")

    # Caminho para o arquivo JSON (nomeado como as fotos) na pasta JSON
    fotos_ponte = [f for f in os.listdir(pasta_ponte) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))]
    for foto in fotos_ponte:
        foto_nome = os.path.splitext(foto)[0]
        json_foto_path = os.path.join(json_dir, f'{foto_nome}.json')

        # Copiar o arquivo JSON base e renomeá-lo para cada foto
        with open(json_base_path, 'r', encoding='utf-8') as json_file:
            json_data = json.load(json_file)

        with open(json_foto_path, 'w', encoding='utf-8') as json_file:
            json.dump(json_data, json_file, indent=4, ensure_ascii=False)

        print(f"Arquivo JSON copiado e renomeado para foto '{foto}' da ponte '{bridge_id_str}': '{json_foto_path}'.")

print("Processo concluído!")
