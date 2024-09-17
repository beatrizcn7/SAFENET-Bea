import pandas as pd

# Caminhos dos arquivos Excel
arquivo_combinacoes_json = 'Combinações JSON 2.xlsx'
arquivo_combinacoes_excel = 'Combinações Excel 2.xlsx'

# Ler os dois arquivos Excel
df_combinacoes_json = pd.read_excel(arquivo_combinacoes_json)
df_combinacoes_excel = pd.read_excel(arquivo_combinacoes_excel)

# Verificar as colunas de ambos para garantir que as colunas sejam consistentes
print("Colunas em Combinações JSON:")
print(df_combinacoes_json.columns)
print("Colunas em Combinações Excel:")
print(df_combinacoes_excel.columns)

# Ajustar os nomes das colunas para garantir consistência (usando 'Qtd' em vez de 'Quantidade')
df_combinacoes_json = df_combinacoes_json[['Tipo de Estrutura', 'Intervalo de Anos', 'Material', 'Quantidade']]
df_combinacoes_excel = df_combinacoes_excel[['Tipo de Estrutura', 'Intervalo de Anos', 'Material', 'Quantidade']]

# Ordenar os DataFrames para garantir que estejam organizados da mesma maneira
df_combinacoes_json = df_combinacoes_json.sort_values(by=['Tipo de Estrutura', 'Intervalo de Anos', 'Material']).reset_index(drop=True)
df_combinacoes_excel = df_combinacoes_excel.sort_values(by=['Tipo de Estrutura', 'Intervalo de Anos', 'Material']).reset_index(drop=True)

# Adicionar um prefixo para diferenciar as tabelas
df_combinacoes_json = df_combinacoes_json.add_prefix('JSON_')
df_combinacoes_excel = df_combinacoes_excel.add_prefix('Excel_')

# Concatenar os dois DataFrames lado a lado
df_comparacao = pd.concat([df_combinacoes_json, df_combinacoes_excel], axis=1)

# Exportar o DataFrame para um novo arquivo Excel
output_excel_path = 'Comparação Simples (JSON vs. Excel) 2.xlsx'
df_comparacao.to_excel(output_excel_path, index=False)

print(f"Arquivo Excel criado: {output_excel_path}")
