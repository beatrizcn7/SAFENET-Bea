import os
import json
import xml.etree.ElementTree as ET
import openpyxl

# Carregar dados do Excel
excel_path = 'Dados.xlsx'  # Substituir pelo caminho correto do arquivo Excel
wb = openpyxl.load_workbook(excel_path, data_only=True)  # data_only=True para obter valores, não fórmulas
sheet = wb['Resultados']  # A segunda folha do arquivo Excel

# Função para buscar dados no Excel por identificador
def buscar_dados_excel(ponte_id):
    for row in sheet.iter_rows(min_row=2, values_only=True):  # Pular a linha de cabeçalho
        if str(row[0]) == ponte_id:
            ano_intervalo = row[13]  # Coluna N (index 13)
            tipo_estrutura = row[14]  # Coluna O (index 14)
            material = row[22]  # Coluna W (index 22)
            return ano_intervalo, tipo_estrutura, material
    return None, None, None

# Validação dos dados JSON
def validar_json(json_path, ponte_id):
    required_fields = ['Id', 'Tipo de Estrutura', 'Intervalo de Anos', 'Material']

    # Garantir a leitura com codificação UTF-8
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

        # Verificar se os campos obrigatórios estão presentes
        for field in required_fields:
            if field not in data:
                print(f'Campo ausente no arquivo {json_path}: {field}')
                return False

        # Buscar dados no Excel
        excel_ano, excel_tipo, excel_material = buscar_dados_excel(ponte_id)

        # Comparar com os dados no Excel
        if (data['Intervalo de Anos'] != excel_ano or
            data['Tipo de Estrutura'] != excel_tipo or
            data['Material'] != excel_material):
            print(f'Discrepância nos dados do arquivo {json_path}')
            print(f'Excel: Ano: {excel_ano}, Tipo: {excel_tipo}, Material: {excel_material}')
            print(f'JSON: Ano: {data["Intervalo de Anos"]}, Tipo: {data["Tipo de Estrutura"]}, Material: {data["Material"]}')
            return False

    return True

# Validação dos dados XML
def validar_xml(xml_path, ponte_id):
    # Garantir que o XML seja lido corretamente
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Definir os campos necessários e procurar na estrutura XML
    try:
        id = root.find('.//Id').text
        tipo_estrutura = root.find('.//TipoDeEstrutura').text
        ano_construcao = root.find('.//IntervaloDeAnos').text
        material = root.find('.//Material').text
    except AttributeError as e:
        # Se algum campo estiver ausente, imprimir a mensagem de erro
        print(f'Campo ausente no arquivo {xml_path}: {str(e)}')
        return False

    # Buscar dados no Excel
    excel_ano, excel_tipo, excel_material = buscar_dados_excel(ponte_id)

    # Comparar com os dados no Excel
    if (ano_construcao != excel_ano or
        tipo_estrutura != excel_tipo or
        material != excel_material):
        print(f'Discrepância nos dados do arquivo {xml_path}')
        print(f'Excel: Ano: {excel_ano}, Tipo: {excel_tipo}, Material: {excel_material}')
        print(f'XML: Ano: {ano_construcao}, Tipo: {tipo_estrutura}, Material: {material}')
        return False

    return True

# Caminhos das pastas
json_root_dir = 'JSON'
xml_root_dir = 'XML'

# Percorrer arquivos JSON
for subdir, _, files in os.walk(json_root_dir):
    ponte_id = os.path.basename(subdir)  # O nome da pasta é o ID da ponte
    for file in files:
        if file.endswith('.json'):
            json_path = os.path.join(subdir, file)
            validar_json(json_path, ponte_id)

# Percorrer arquivos XML
for subdir, _, files in os.walk(xml_root_dir):
    ponte_id = os.path.basename(subdir)  # O nome da pasta é o ID da ponte
    for file in files:
        if file.endswith('.xml'):
            xml_path = os.path.join(subdir, file)
            validar_xml(xml_path, ponte_id)
