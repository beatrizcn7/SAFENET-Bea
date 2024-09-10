import os
import pandas as pd
import xml.etree.ElementTree as ET

# Caminho para o ficheiro Excel e diretórios
excel_path = 'Dados.xlsx'
diretorio_principal = 'Fotos'
diretorio_xml = 'XML'

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

    # Cria XMLs apenas para as fotos
    for foto in fotos_ponte:
        foto_nome = os.path.splitext(foto)[0]
        xml_foto_path = os.path.join(diretorio_xml, bridge_id_str, f'{foto_nome}.xml')

        # Garante que o diretório para os XMLs existe
        os.makedirs(os.path.dirname(xml_foto_path), exist_ok=True)

        # Cria a estrutura XML
        ponte_element = ET.Element('Ponte')

        id_element = ET.SubElement(ponte_element, 'Id')
        id_element.text = str(bridge_id_int)

        tipo_estrutura_element = ET.SubElement(ponte_element, 'TipoDeEstrutura')
        tipo_estrutura_element.text = tipo_estrutura

        intervalo_anos_element = ET.SubElement(ponte_element, 'IntervaloDeAnos')
        intervalo_anos_element.text = intervalo_anos

        material_element = ET.SubElement(ponte_element, 'Material')
        material_element.text = material

        # Gera a árvore XML
        tree = ET.ElementTree(ponte_element)

        # Escreve o arquivo XML
        tree.write(xml_foto_path, encoding='utf-8', xml_declaration=True)

        print(f"XML criado para '{foto}' da ponte '{bridge_id_str}': '{xml_foto_path}'.")

print("!!CONCLUÍDO!!")
