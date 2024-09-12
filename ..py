import pandas as pd

# Caminhos dos arquivos Excel
arquivo_combinacoes_completas_json = 'Combinações Completas JSON.xlsx'
arquivo_combinacoes_completas_excel = 'Combinações Completas Excel.xlsx'

# Ler os dois arquivos Excel
df_combinacoes_completas_json = pd.read_excel(arquivo_combinacoes_completas_json)
df_combinacoes_completas_excel = pd.read_excel(arquivo_combinacoes_completas_excel)

# Exibir as colunas para garantir que estamos usando os nomes corretos
print("Colunas em Combinações Completas JSON:")
print(df_combinacoes_completas_json.columns)
print("Colunas em Combinações Completas Excel:")
print(df_combinacoes_completas_excel.columns)

# Ajustar os nomes das colunas para garantir consistência
# Ajustar as colunas para garantir que tenham as mesmas colunas
df_combinacoes_completas_json = df_combinacoes_completas_json.rename(columns={
    'Ano de Construção': 'Intervalo de Anos'
})

# Verificar se as colunas estão corretas agora
print("Colunas ajustadas em Combinações Completas JSON:")
print(df_combinacoes_completas_json.columns)

# Selecionar apenas as colunas relevantes
df_combinacoes_completas_json = df_combinacoes_completas_json[['Intervalo de Anos', 'Tipo de Estrutura', 'Material', 'Quantidade']]
df_combinacoes_completas_excel = df_combinacoes_completas_excel[['Intervalo de Anos', 'Tipo de Estrutura', 'Material', 'Quantidade']]

# Ordenar os DataFrames para garantir que estejam na mesma ordem
df_combinacoes_completas_json = df_combinacoes_completas_json.sort_values(by=['Intervalo de Anos', 'Tipo de Estrutura', 'Material']).reset_index(drop=True)
df_combinacoes_completas_excel = df_combinacoes_completas_excel.sort_values(by=['Intervalo de Anos', 'Tipo de Estrutura', 'Material']).reset_index(drop=True)

# Adicionar um prefixo para diferenciar as tabelas
df_combinacoes_completas_json = df_combinacoes_completas_json.add_prefix('JSON_')
df_combinacoes_completas_excel = df_combinacoes_completas_excel.add_prefix('Excel_')

# Concatenar os dois DataFrames lado a lado
df_comparacao = pd.concat([df_combinacoes_completas_json, df_combinacoes_completas_excel], axis=1)

# Exportar o DataFrame para um novo arquivo Excel
output_excel_path = 'Comparação Completas (JSON vs. Excel).xlsx'
df_comparacao.to_excel(output_excel_path, index=False)

print(f"Arquivo Excel criado: {output_excel_path}")
