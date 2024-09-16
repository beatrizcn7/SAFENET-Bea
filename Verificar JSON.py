import os
import json

# Caminho da pasta de ficheiros JSON
json_root_dir = 'JSON'

# Campos obrigatórios
required_fields = ['Id', 'Tipo de Estrutura', 'Material', 'Intervalo de Anos']


# Função para verificar campos obrigatórios em JSON
def verificar_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as file:
        try:
            data = json.load(file)
        except json.JSONDecodeError:
            print(f'Erro ao ler o arquivo JSON {json_path}')
            return False

        # Verificar se os campos obrigatórios estão presentes
        missing_fields = [field for field in required_fields if field not in data]

        if missing_fields:
            print(f'Campos ausentes no arquivo {json_path}: {", ".join(missing_fields)}')
            return False
        return True


# Percorrer todos os arquivos JSON na pasta
for subdir, _, files in os.walk(json_root_dir):
    for file in files:
        if file.endswith('.json'):
            json_path = os.path.join(subdir, file)
            verificar_json(json_path)
