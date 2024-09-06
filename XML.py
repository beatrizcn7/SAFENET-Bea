import os
import pandas as pd
import xml.etree.ElementTree as ET

excel_path = 'Dados.xlsx'
diretorio_principal = 'Fotos'
diretorio_xml = 'XML'

try:
    df = pd.read_excel(excel_path, sheet_name='Resultados')
except FileNotFoundError:
    print(f"Erro: '{excel_path}' não encontrado.")
    exit(1)
except Exception as e:
    print(f"Erro ao ler '{excel_path}': {e}")
    exit(1)

if 'Id' not in df.columns:
    print(f"Erro: Coluna 'Id' não encontrada")
    exit(1)

for bridge_id_int in range(1, 2557):
    bridge_id_str = str(bridge_id_int)
    pasta_ponte = os.path.join(diretorio_principal, bridge_id_str)

    if not os.path.isdir(pasta_ponte):
        print(f"Aviso: Pasta não encontrada para ponte '{bridge_id_str}'")
        continue

    try:
        linha_ponte = df[df['Id'] == bridge_id_int].iloc[0]
    except IndexError:
        print(f"Aviso: Id '{bridge_id_str}' não encontrado")
        continue

    tipo_estrutura = linha_ponte['TEC - Tipo de Estrutura']
    intervalo_anos = linha_ponte['Intervalo de Anos']
    material = linha_ponte['CON - Material 1']

    root = ET.Element("Ponte")
    ET.SubElement(root, "Id").text = str(bridge_id_int)
    ET.SubElement(root, "Tipo_de_Estrutura").text = tipo_estrutura
    ET.SubElement(root, "Intervalo_de_Anos").text = intervalo_anos
    ET.SubElement(root, "Material").text = material

    xml_dir = os.path.join(diretorio_xml, bridge_id_str)
    os.makedirs(xml_dir, exist_ok=True)

    xml_base_path = os.path.join(xml_dir, f'{bridge_id_str}.xml')

    tree = ET.ElementTree(root)
    tree.write(xml_base_path, encoding='utf-8', xml_declaration=True)

    print(f"XML criado: ponte '{bridge_id_str}': '{xml_base_path}'.")

    fotos_ponte = [f for f in os.listdir(pasta_ponte) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))]
    for foto in fotos_ponte:
        foto_nome = os.path.splitext(foto)[0]
        xml_foto_path = os.path.join(xml_dir, f'{foto_nome}.xml')

        tree = ET.ElementTree(root)
        tree.write(xml_foto_path, encoding='utf-8', xml_declaration=True)

        print(f"XML para '{foto}' da ponte '{bridge_id_str}': '{xml_foto_path}'.")

print("!!CONCLUÍDO!!")
