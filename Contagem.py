import os
import json
import pandas as pd
from collections import Counter

# Diretório onde estão os JSONs
diretorio_json = 'JSON'

# Dicionário para armazenar as combinações
combinacoes_counter = Counter()

# Percorrer todas as subpastas e arquivos JSON
for root, dirs, files in os.walk(diretorio_json):
    for file in files:
        if file.endswith('.json'):
            json_path = os.path.join(root, file)

            # Ler o arquivo JSON
            with open(json_path, 'r', encoding='utf-8') as json_file:
                json_data = json.load(json_file)

            # Extrair as informações
            tipo_estrutura = json_data.get('Tipo de Estrutura')
            intervalo_anos = json_data.get('Intervalo de Anos')
            material = json_data.get('Material')

            # Criar uma combinação como uma tupla
            combinacao = (tipo_estrutura, intervalo_anos, material)

            # Contabilizar a combinação
            combinacoes_counter[combinacao] += 1

# Criar um DataFrame a partir das combinações e contagens
combinacoes_df = pd.DataFrame(
    combinacoes_counter.items(),
    columns=['Combinação', 'Quantidade']
)

# Separar as combinações em colunas distintas
combinacoes_df[['Tipo de Estrutura', 'Intervalo de Anos', 'Material']] = pd.DataFrame(combinacoes_df['Combinação'].tolist(), index=combinacoes_df.index)

# Remover a coluna 'Combinação' pois já separamos em colunas distintas
combinacoes_df = combinacoes_df.drop(columns=['Combinação'])

# Reorganizar as colunas para uma ordem mais clara
combinacoes_df = combinacoes_df[['Tipo de Estrutura', 'Intervalo de Anos', 'Material', 'Quantidade']]

# Exportar para um arquivo Excel
output_excel_path = 'Combinações.xlsx'
combinacoes_df.to_excel(output_excel_path, index=False)

print(f"Arquivo Excel criado: {output_excel_path}")
