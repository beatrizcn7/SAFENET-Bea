import pandas as pd
from collections import Counter
import itertools

# Caminho do Excel 'Base de Dados'
excel_base_dados_path = 'Base de Dados.xlsx'

# Ler o arquivo Excel 'Base de Dados', especificamente a segunda folha
df_base = pd.read_excel(excel_base_dados_path, sheet_name=1)  # 'sheet_name=1' refere-se à segunda folha (conta a partir de 0)

# Verificar os nomes das colunas para garantir que estamos usando os corretos
print("Colunas no DataFrame:")
print(df_base.columns)

# Criar um DataFrame com as colunas relevantes
# Ajuste os nomes das colunas de acordo com o que você tiver no Excel
df_combinacoes = df_base[['TEC - Tipo de Estrutura 2', 'Intervalo de Anos', 'CON - Material', 'Fotos']].dropna()

# Renomear colunas para simplificar a manipulação
df_combinacoes.columns = ['Tipo de Estrutura', 'Intervalo de Anos', 'Material', 'Quantidade']

# Criar uma combinação para cada linha e multiplicar pela quantidade de fotos
combinacoes_counter = Counter()

for _, row in df_combinacoes.iterrows():
    combinacao = (row['Tipo de Estrutura'], row['Intervalo de Anos'], row['Material'])
    quantidade_fotos = row['Quantidade']  # 'Quantidade' contém o valor das fotos
    combinacoes_counter[combinacao] += quantidade_fotos

# Listas com os atributos fornecidos
intervalos = ['Antes de 1960', '1960-1983', '1983-2010', 'Depois de 2010']
tipos_estrutura = [
    'Arco', 'Outro', 'Pórtico', 'Quadro', 'Tabuleiro simples/apoiado', 'Vãos Multiplos'
]
materiais = [
    'Aço', 'Alvenaria de Pedra', 'Alvenaria de Tijolo', 'Betão Armado', 'Outros'
]

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
output_excel_path = 'Combinações Completas Excel 2.xlsx'
df_combinacoes.to_excel(output_excel_path, index=False)

print(f"Arquivo Excel criado: {output_excel_path}")
