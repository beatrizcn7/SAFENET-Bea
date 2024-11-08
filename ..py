import pandas as pd
from collections import Counter

# Caminho do Excel 'Base de Dados'
excel_base_dados_path = 'Base de Dados.xlsx'

# Ler o arquivo Excel 'Base de Dados', especificamente a segunda folha
df_base = pd.read_excel(excel_base_dados_path, sheet_name=1)  # 'sheet_name=1' refere-se à segunda folha (conta a partir de 0)

# Verificar os nomes das colunas para garantir que estamos usando os corretos
print("Colunas no DataFrame:")
print(df_base.columns)

# Criar um DataFrame com as colunas relevantes
# Ajuste os nomes das colunas de acordo com o que você tiver no Excel
df_combinacoes = df_base[['TEC - Tipo de Estrutura 3', 'Intervalo de Anos 1', 'Material', 'Fotos']].dropna()

# Renomear colunas para simplificar a manipulação
df_combinacoes.columns = ['Tipo de Estrutura', 'Intervalo de Anos', 'Material', 'Quantidade']

# Criar uma combinação para cada linha e multiplicar pela quantidade de fotos
combinacoes_counter = Counter()

for _, row in df_combinacoes.iterrows():
    combinacao = (row['Tipo de Estrutura'], row['Intervalo de Anos'], row['Material'])
    quantidade_fotos = row['Quantidade']  # 'Quantidade' contém o valor das fotos
    combinacoes_counter[combinacao] += quantidade_fotos

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
output_excel_path = 'Combinações 1.xlsx'
combinacoes_df.to_excel(output_excel_path, index=False)

print(f"Arquivo Excel criado: {output_excel_path}")
