import os
from collections import Counter


def verificar_duplicados(pasta_final):
    arquivos = os.listdir(pasta_final)
    contagem = Counter(arquivos)
    duplicados = {arquivo: qtd for arquivo, qtd in contagem.items() if qtd > 1}

    return duplicados


def main():
    pasta_final = 'Pasta Final'  # Substitua pelo caminho da pasta final
    duplicados = verificar_duplicados(pasta_final)

    if duplicados:
        print("Arquivos duplicados encontrados:")
        for arquivo, qtd in duplicados.items():
            print(f"{arquivo}: {qtd} vezes")
    else:
        print("Nenhum arquivo duplicado encontrado.")


if __name__ == "__main__":
    main()
