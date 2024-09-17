import os
import xml.etree.ElementTree as ET


def verificar_anotacoes(pasta_final):
    imagens_sem_anotacao = []

    # Obtém a lista de arquivos .jpg e .xml
    arquivos = os.listdir(pasta_final)
    arquivos_jpg = {f for f in arquivos if f.endswith('.jpg')}
    arquivos_xml = {f for f in arquivos if f.endswith('.xml')}

    # Cria um conjunto de nomes baseados em arquivos .jpg
    nomes_base_jpg = {os.path.splitext(f)[0] for f in arquivos_jpg}

    # Verifica se há arquivos .xml para cada imagem .jpg e se o arquivo XML contém anotações
    for nome_base in nomes_base_jpg:
        xml_correspondente = f"{nome_base}.xml"
        if xml_correspondente not in arquivos_xml:
            imagens_sem_anotacao.append(f"{nome_base}.jpg")
        else:
            caminho_xml = os.path.join(pasta_final, xml_correspondente)
            if not verificar_se_anotado(caminho_xml):
                imagens_sem_anotacao.append(f"{nome_base}.jpg")

    return imagens_sem_anotacao


def verificar_se_anotado(caminho_xml):
    """Verifica se o arquivo XML contém anotações válidas (tag <annotation>)."""
    try:
        tree = ET.parse(caminho_xml)
        root = tree.getroot()

        # Verifica se a tag principal é <annotation>
        if root.tag == "annotation":
            # Se for um arquivo anotado, retorna True
            return True
    except ET.ParseError:
        # Se houver erro ao parsear o XML, considera que não está anotado
        return False

    # Se não encontrar a tag <annotation>, o arquivo XML não é considerado anotado
    return False


def main():
    pasta_final = 'Pasta Final'  # Substitua pelo caminho para a pasta onde você armazenou as imagens e os arquivos XML

    imagens_sem_anotacao = verificar_anotacoes(pasta_final)

    if imagens_sem_anotacao:
        print("Imagens sem anotação:")
        for imagem in imagens_sem_anotacao:
            print(imagem)
        print(f"\nNúmero total de imagens sem anotação: {len(imagens_sem_anotacao)}")
    else:
        print("Todas as imagens possuem anotações válidas.")


if __name__ == "__main__":
    main()
