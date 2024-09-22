import os
import xml.etree.ElementTree as ET

# Caminhos das pastas
pasta_final = 'Pasta Final'
pasta_xml2 = 'XML 2'

# Função para extrair os atributos do ficheiro XML dentro da pasta XML 2
def extrair_atributos_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Extrair informações do ficheiro XML da pasta XML 2
    tipo_estrutura = root.find('TipoDeEstrutura').text
    intervalo_anos = root.find('IntervaloDeAnos').text
    material = root.find('Material').text

    return {
        'tipo_estrutura': tipo_estrutura,
        'intervalo_anos': intervalo_anos,
        'material': material
    }

# Função para adicionar atributos ao XML na Pasta Final
def adicionar_atributos_ao_xml(xml_file, atributos_imagem):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Criar uma nova tag <structure> para os atributos
    structure = ET.SubElement(root, "structure")
    tipo = ET.SubElement(structure, "type")
    tipo.text = atributos_imagem['tipo_estrutura']
    material = ET.SubElement(structure, "material")
    material.text = atributos_imagem['material']
    ano = ET.SubElement(structure, "year_range")
    ano.text = atributos_imagem['intervalo_anos']

    # Salvar o XML modificado com a codificação UTF-8
    tree.write(xml_file, encoding='utf-8', xml_declaration=True)

# Percorrer a Pasta Final e adicionar os atributos correspondentes da XML 2
for filename in os.listdir(pasta_final):
    if filename.endswith('.xml'):
        # Obter o número identificador da ponte e a posição (por exemplo, '1' e '1' de '1_1.xml')
        identificador_ponte, _ = filename.split('_')

        # Caminho da pasta correspondente dentro da XML 2
        xml2_subpasta = os.path.join(pasta_xml2, identificador_ponte)

        # Verificar se a subpasta correspondente existe
        if os.path.isdir(xml2_subpasta):
            # Caminho completo para o ficheiro XML dentro da subpasta
            xml2_file_path = os.path.join(xml2_subpasta, filename)

            if os.path.exists(xml2_file_path):
                # Extrair os atributos do ficheiro XML da XML 2
                atributos = extrair_atributos_xml(xml2_file_path)

                # Caminho do ficheiro XML na Pasta Final
                xml_final_path = os.path.join(pasta_final, filename)

                # Adicionar os atributos ao XML da Pasta Final
                adicionar_atributos_ao_xml(xml_final_path, atributos)
                print(f"Atributos adicionados ao ficheiro: {filename}")
            else:
                print(f"Ficheiro XML correspondente não encontrado na pasta: {xml2_subpasta}")
        else:
            print(f"Subpasta correspondente não encontrada: {xml2_subpasta}")
