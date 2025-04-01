# Beatriz Neves, 13 de setembro de 2024
# Input = XML
# Output = Informação no terminal
# Objetivo = Verifica se não informação em falta nos ficheiros XML


import os
import xml.etree.ElementTree as ET

# Caminho da pasta de ficheiros XML
xml_root_dir = 'XML'

# Campos obrigatórios
required_fields = ['Id', 'TipoDeEstrutura', 'Material', 'IntervaloDeAnos']


# Função para verificar campos obrigatórios em XML
def verificar_xml(xml_path):
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Verificar se os campos obrigatórios estão presentes
        missing_fields = [field for field in required_fields if root.find(field) is None]

        if missing_fields:
            print(f'Campos ausentes no arquivo {xml_path}: {", ".join(missing_fields)}')
            return False
        return True

    except ET.ParseError:
        print(f'Erro ao ler o arquivo XML {xml_path}')
        return False


# Percorrer todos os arquivos XML na pasta
for subdir, _, files in os.walk(xml_root_dir):
    for file in files:
        if file.endswith('.xml'):
            xml_path = os.path.join(subdir, file)
            verificar_xml(xml_path)
