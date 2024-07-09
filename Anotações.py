import os
import pandas as pd
import json
import xml.etree.ElementTree as ET
import shutil

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

    # Caminho para o arquivo JSON na subpasta específica
    json_filename = f'{bridge_id_str}.json'
    json_path = os.path.join(diretorio_json, bridge_id_str, json_filename)

    # Criar diretório para os arquivos JSON se não existir
    os.makedirs(os.path.dirname(json_path), exist_ok=True)

    # Escrever as informações da ponte no arquivo JSON (criando apenas uma vez)
    with open(json_path, 'w', encoding='utf-8') as json_file:
        json.dump(ponte_info, json_file, indent=4, ensure_ascii=False)

    print(f"Arquivo JSON criado para ponte '{bridge_id_str}': '{json_path}'.")

    # Caminho para o arquivo XML (nomeado como as fotos) na pasta principal
    fotos_ponte = [f for f in os.listdir(pasta_ponte) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))]
    for foto in fotos_ponte:
        foto_path = os.path.join(pasta_ponte, foto)
        xml_filename = os.path.splitext(foto)[0] + '.xml'
        xml_foto_path = os.path.join(pasta_ponte, xml_filename)

        # Criar ou atualizar o arquivo XML com as informações da ponte
        root = ET.Element('ponte')
        ET.SubElement(root, 'tipo').text = str(tipo_estrutura)
        ET.SubElement(root, 'intervalo_anos').text = str(intervalo_anos)
        ET.SubElement(root, 'material').text = str(material)

        tree = ET.ElementTree(root)
        tree.write(xml_foto_path)

        print(f"Arquivo XML criado/atualizado para foto '{foto}' da ponte '{bridge_id_str}': '{xml_foto_path}'.")

    # Copiar o arquivo JSON para cada foto na subpasta
    for foto in fotos_ponte:
        json_foto_path = os.path.join(pasta_ponte, f'{bridge_id_str}.json')
        shutil.copy(json_path, json_foto_path)
        print(f"Arquivo JSON copiado para foto '{foto}' da ponte '{bridge_id_str}': '{json_foto_path}'.")

print("Processo concluído!")
