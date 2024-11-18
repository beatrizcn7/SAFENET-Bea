import os
import xml.etree.ElementTree as ET
from xml.dom import minidom

# Caminho da pasta final com os XML e fotos
pasta_final = "Pasta Final"
novo_tamanho = (224, 224, 3)  # Largura, altura, profundidade (RGB)

# Função para processar cada arquivo XML
def converter_xml(xml_path, image_filename):
    # Ler o XML original
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Extrair informações do XML original
    id_ponte = root.find("Id").text
    tipo_estrutura = root.find("TipoDeEstrutura").text
    intervalo_anos = root.find("IntervaloDeAnos").text
    material = root.find("Material").text

    # Criar a nova estrutura XML
    annotation = ET.Element("annotation", verified="yes")

    folder = ET.SubElement(annotation, "folder")
    folder.text = "Pasta Final"

    filename = ET.SubElement(annotation, "filename")
    filename.text = image_filename

    path = ET.SubElement(annotation, "path")
    image_path = os.path.join(pasta_final, image_filename)
    path.text = image_path

    source = ET.SubElement(annotation, "source")
    database = ET.SubElement(source, "database")
    database.text = "Unknown"

    size = ET.SubElement(annotation, "size")
    width = ET.SubElement(size, "width")
    width.text = str(novo_tamanho[0])
    height = ET.SubElement(size, "height")
    height.text = str(novo_tamanho[1])
    depth = ET.SubElement(size, "depth")
    depth.text = str(novo_tamanho[2])

    segmented = ET.SubElement(annotation, "segmented")
    segmented.text = "0"

    # Adicionar a estrutura
    structure = ET.SubElement(annotation, "structure")

    type_elem = ET.SubElement(structure, "type")
    type_elem.text = tipo_estrutura

    material_elem = ET.SubElement(structure, "material")
    material_elem.text = material

    year_range_elem = ET.SubElement(structure, "year_range")
    year_range_elem.text = intervalo_anos

    # Formatar o XML com indentação e quebras de linha
    xml_str = ET.tostring(annotation, encoding='utf-8')
    formatted_xml = minidom.parseString(xml_str).toprettyxml(indent="    ")

    # Salvar o novo XML
    novo_xml_path = os.path.join(pasta_final, f"{os.path.splitext(image_filename)[0]}.xml")
    with open(novo_xml_path, "w", encoding='utf-8') as f:
        f.write(formatted_xml)
    print(f"Arquivo XML salvo: {novo_xml_path}")


# Processar todos os arquivos XML na pasta
for filename in os.listdir(pasta_final):
    if filename.endswith(".xml"):
        xml_path = os.path.join(pasta_final, filename)
        # O nome da imagem correspondente é o mesmo que o XML, mas com extensão .jpg
        image_filename = f"{os.path.splitext(filename)[0]}.jpg"
        if os.path.exists(os.path.join(pasta_final, image_filename)):
            converter_xml(xml_path, image_filename)
        else:
            print(f"Imagem não encontrada para o XML: {xml_path}")
