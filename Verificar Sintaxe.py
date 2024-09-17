import os
import json

# Definir os valores esperados para cada campo, usando aspas consistentes
valid_id_range = range(1, 2557)
valid_intervalo_anos = ["Antes de 1960", "1960-1983", "1983-2010", "Depois de 2010"]
valid_tipo_estrutura = ["Arco", "Outro", "Pórtico", "Quadro", "Tabuleiro simples/apoiado", "Vãos Multiplos"]
valid_material = ["Aço", "Alvenaria de Pedra", "Alvenaria de Tijolo", "Betão Armado", "Outros"]



# Função para normalizar e verificar um arquivo JSON
def verificar_e_normalizar_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as file:
        try:
            data = json.load(file)
        except json.JSONDecodeError:
            print(f'Erro ao ler o arquivo JSON {json_path}')
            return False

        # Verificar e corrigir o ID
        if not isinstance(data.get('Id'), int) or data['Id'] not in valid_id_range:
            print(f'Id inválido no arquivo {json_path}: {data.get("Id")}')
            return False

        # Verificar e normalizar Intervalo de Anos
        intervalo_anos = data.get('Intervalo de Anos')
        if intervalo_anos not in valid_intervalo_anos:
            print(f'Intervalo de Anos inválido no arquivo {json_path}: {intervalo_anos}')
            return False

        # Verificar e normalizar Tipo de Estrutura
        tipo_estrutura = data.get('Tipo de Estrutura')
        if tipo_estrutura not in valid_tipo_estrutura:
            print(f'Tipo de Estrutura inválido no arquivo {json_path}: {tipo_estrutura}')
            return False

        # Verificar e normalizar Material
        material = data.get('Material')
        if material not in valid_material:
            print(f'Material inválido no arquivo {json_path}: {material}')
            return False

        # Se tudo está correto, retornar True
        return True


# Caminho da pasta de ficheiros JSON
json_root_dir = 'JSON 2'

# Percorrer todos os arquivos JSON na pasta
for subdir, _, files in os.walk(json_root_dir):
    for file in files:
        if file.endswith('.json'):
            json_path = os.path.join(subdir, file)
            verificar_e_normalizar_json(json_path)
