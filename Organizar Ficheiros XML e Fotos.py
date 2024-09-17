import os
import shutil
import re

def criar_pasta_final(pasta_final):
    """Cria a pasta final se ela não existir."""
    if not os.path.exists(pasta_final):
        os.makedirs(pasta_final)

def mover_arquivos(pasta_origem, pasta_final):
    """Move todos os arquivos XML e imagens para a pasta final."""
    for pasta_raiz, _, arquivos in os.walk(pasta_origem):
        for arquivo in arquivos:
            if arquivo.endswith('.xml') or arquivo.endswith('.jpg') or arquivo.endswith('.png'):
                caminho_completo = os.path.join(pasta_raiz, arquivo)
                shutil.copy2(caminho_completo, pasta_final)

def organizar_pasta_final(pasta_final):
    """Organiza os arquivos na pasta final em ordem crescente de identificação e posição."""
    arquivos = os.listdir(pasta_final)

    def extrair_identificacao_e_posicao(nome_arquivo):
        """Extrai identificação e posição de um nome de arquivo para ordenação."""
        match = re.match(r'(\d+)_([\d]+)(.*)', nome_arquivo)
        if match:
            identificacao = int(match.group(1))
            posicao = int(match.group(2))
            return (identificacao, posicao)
        return (float('inf'), float('inf'))

    # Ordena os arquivos numericamente com base na identificação e posição.
    arquivos.sort(key=lambda x: extrair_identificacao_e_posicao(x))

    for i, arquivo in enumerate(arquivos):
        caminho_antigo = os.path.join(pasta_final, arquivo)
        novo_nome = arquivo  # Não altera o nome, apenas organiza
        caminho_novo = os.path.join(pasta_final, novo_nome)
        os.rename(caminho_antigo, caminho_novo)

def main():
    pasta_imagens = 'Fotos Normalizadas'
    pasta_xml = 'XML 2'
    pasta_final = 'Pasta Final'

    criar_pasta_final(pasta_final)

    # Mover arquivos de imagens
    mover_arquivos(pasta_imagens, pasta_final)

    # Mover arquivos XML
    mover_arquivos(pasta_xml, pasta_final)

    # Organizar arquivos na pasta final
    organizar_pasta_final(pasta_final)

if __name__ == "__main__":
    main()
