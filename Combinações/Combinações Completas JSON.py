import os
import json
import pandas as pd
from collections import Counter
import itertools

# Listas com os atributos fornecidos
intervalos = ['Antes de 1960', '1960-1983', '1983-2010', 'Depois de 2010']
tipos_estrutura = [
    'Arco', 'Outro', 'Pórtico', 'Quadro', 'Tabuleiro simples/apoiado', 'Vãos Multiplos'
]
materiais = [
    'Aço', 'Alvenaria de Pedra', 'Alvenaria de Tijolo', 'Betão Armado', 'Outros'
]


# Diretório onde estão os arquivos JSON
diretorio_json = 'JSON 2'

# Dicionário para armazenar as combinações encontradas nos arquivos JSON
combinacoes_counter = Counter()

# Percorrer todos os arquivos JSON no diretório
for root, dirs, files in os.walk(diretorio_json):
    for file in files:
        if file.endswith('.json'):
            json_path = os.path.join(root, file)

            # Ler o arquivo JSON
            with open(json_path, 'r', encoding='utf-8') as json_file:
                json_data = json.load(json_file)

            # Extrair os atributos
            tipo_estrutura = json_data.get('Tipo de Estrutura')
            intervalo_anos = json_data.get('Intervalo de Anos')
            material = json_data.get('Material')

            # Criar uma combinação e contar sua ocorrência
            combinacao = (tipo_estrutura, intervalo_anos, material)
            combinacoes_counter[combinacao] += 1

# Gerar todas as combinações possíveis
todas_combinacoes = list(itertools.product(tipos_estrutura, intervalos, materiais))

# Preparar uma lista para armazenar os resultados
resultado_combinacoes = []

# Para cada combinação possível, verificar se está no JSON e adicionar a contagem
for combinacao in todas_combinacoes:
    qtd = combinacoes_counter.get(combinacao, 0)  # Se não estiver no JSON, a quantidade é 0
    resultado_combinacoes.append({
        'Tipo de Estrutura': combinacao[0],
        'Intervalo de Anos': combinacao[1],
        'Material': combinacao[2],
        'Quantidade': qtd
    })

# Criar um DataFrame a partir dos resultados
df_combinacoes = pd.DataFrame(resultado_combinacoes)

# Exportar o DataFrame para um arquivo Excel
output_excel_path = 'Combinações Completas JSON 2.xlsx'
df_combinacoes.to_excel(output_excel_path, index=False)

print(f"Arquivo Excel criado: {output_excel_path}")
