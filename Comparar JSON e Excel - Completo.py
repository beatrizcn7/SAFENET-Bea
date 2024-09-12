import pandas as pd

# Caminhos dos arquivos Excel
arquivo_combinacoes_json = 'Combinações Completas JSON.xlsx'
arquivo_combinacoes_excel = 'Combinações Completas Excel.xlsx'

# Ler os dois arquivos Excel
df_combinacoes_json = pd.read_excel(arquivo_combinacoes_json)
df_combinacoes_excel = pd.read_excel(arquivo_combinacoes_excel)

# Exibir as colunas para garantir que estamos usando os nomes corretos
print("Colunas em Combinações JSON:")
print(df_combinacoes_json.columns)
print("Colunas em Combinações Completas:")
print(df_combinacoes_excel.columns)

# Ajustar os nomes das colunas para garantir consistência
# Aqui, o nome da coluna para a quantidade é 'Qtd' em JSON e 'Quantidade' em Excel
# Vamos renomear essas colunas para o mesmo nome
df_combinacoes_json.rename(columns={'Qtd': 'Quantidade'}, inplace=True)
df_combinacoes_excel.rename(columns={'Quantidade': 'Quantidade'}, inplace=True)

# Ordenar os DataFrames para garantir que estejam na mesma ordem
df_combinacoes_json = df_combinacoes_json.sort_values(by=['Tipo de Estrutura', 'Intervalo de Anos', 'Material']).reset_index(drop=True)
df_combinacoes_excel = df_combinacoes_excel.sort_values(by=['Tipo de Estrutura', 'Intervalo de Anos', 'Material']).reset_index(drop=True)

# Adicionar um prefixo para diferenciar as tabelas
df_combinacoes_json = df_combinacoes_json.add_prefix('JSON_')
df_combinacoes_excel = df_combinacoes_excel.add_prefix('Excel_')

# Concatenar os dois DataFrames lado a lado
df_comparacao = pd.concat([df_combinacoes_json, df_combinacoes_excel], axis=1)

# Exportar o DataFrame para um novo arquivo Excel
output_excel_path = 'Comparação Completas (JSON vs. Excel).xlsx'
df_comparacao.to_excel(output_excel_path, index=False)

print(f"Arquivo Excel criado: {output_excel_path}")
