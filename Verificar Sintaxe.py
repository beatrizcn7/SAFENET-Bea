# Beatriz Neves, 13 de setembro de 2024
# Input = XML
# Output = Informação no terminal
# Objetivo = Verifica que não há nenhum erro com os ficheiros XML


import os
import xml.etree.ElementTree as ET

# Definir os valores esperados para cada campo
valid_id_range = range(1, 2557)
valid_intervalo_anos = ["Antes de 1960", "1960-1983", "1983-2010", "Depois de 2010"]
valid_tipo_estrutura = ["Arco", "Outro", "Pórtico", "Quadro", "Tabuleiro simples/apoiado", "Vãos Multiplos"]
valid_material = ["Aço", "Alvenaria de Pedra", "Alvenaria de Tijolo", "Betão Armado", "Outros"]

# Função para verificar e normalizar um arquivo XML
def verificar_e_normalizar_xml(xml_path):
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Verificar e corrigir o ID
        id_element = root.find('Id')
        if id_element is None or not id_element.text.isdigit() or int(id_element.text) not in valid_id_range:
            print(f'Id inválido no ficheiro {xml_path}: {id_element.text if id_element is not None else "None"}')
            return False

        # Verificar e normalizar Intervalo de Anos
        intervalo_anos_element = root.find('IntervaloDeAnos')
        intervalo_anos = intervalo_anos_element.text if intervalo_anos_element is not None else None
        if intervalo_anos not in valid_intervalo_anos:
            print(f'Intervalo de Anos inválido no ficheiro {xml_path}: {intervalo_anos}')
            return False

        # Verificar e normalizar Tipo de Estrutura
        tipo_estrutura_element = root.find('TipoDeEstrutura')
        tipo_estrutura = tipo_estrutura_element.text if tipo_estrutura_element is not None else None
        if tipo_estrutura not in valid_tipo_estrutura:
            print(f'Tipo de Estrutura inválido no ficheiro {xml_path}: {tipo_estrutura}')
            return False

        # Verificar e normalizar Material
        material_element = root.find('Material')
        material = material_element.text if material_element is not None else None
        if material not in valid_material:
            print(f'Material inválido no ficheiro {xml_path}: {material}')
            return False

        # Se tudo está correto, retornar True
        return True

    except ET.ParseError:
        print(f'Erro ao ler o ficheiro XML {xml_path}')
        return False


# Caminho da pasta de ficheiros XML
xml_root_dir = 'XML'

# Percorrer todos os ficheiros XML na pasta
for subdir, _, files in os.walk(xml_root_dir):
    for file in files:
        if file.endswith('.xml'):
            xml_path = os.path.join(subdir, file)
            verificar_e_normalizar_xml(xml_path)
