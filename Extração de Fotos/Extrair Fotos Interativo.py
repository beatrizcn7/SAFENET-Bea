# Beatriz Neves, 9 de agosto de 2024
# Input = Informação no terminal
# Output = Fotos
# Objetivo = Ajuda a guardar fotos com os URLs de Google Street View


# Resquests = fazer requisições HTTP em Python
import requests
# Os = criação de pastas, neste caso
import os
# Unquote = descodificar strings de URLs
from urllib.parse import unquote

def extrair_parametros(url):
    # parametros é um dicionário vazio onde vão ficar os valores extraídos do URL
    parametros = {}
    # unquote descodifica o URL
    url_decoded = unquote(url)

    try:
        # Procurar o início da string "panoid=" no URL, panoid é id panorâmico que identifica a imagem panorâmica específica
        # Cada imagem no Street View tem um panoid único
        panoid_start = url_decoded.find("panoid=") + len("panoid=")
        panoid_end = url_decoded.find("&", panoid_start)
        parametros['panoid'] = url_decoded[panoid_start:panoid_end] # Valor de panoid adicionado ao dicionário

        # Procurar o valor de yaw, rotação horizontal da câmara, usa a mesma lógica que a anterior
        yaw_start = url_decoded.find("yaw=") + len("yaw=")
        yaw_end = url_decoded.find("&", yaw_start)
        if yaw_end == -1:
            yaw_end = len(url_decoded)
        parametros['yaw'] = url_decoded[yaw_start:yaw_end].split("!")[0] # Valor de yaw adicionado ao dicionário

        # Procurar o valor de pitch, ângulo de inclinação da câmara, usa a mesma lógica que a anterior
        pitch_start = url_decoded.find("pitch=") + len("pitch=")
        pitch_end = url_decoded.find("&", pitch_start)
        if pitch_end == -1:
            pitch_end = len(url_decoded)
        parametros['pitch'] = url_decoded[pitch_start:pitch_end].split("!")[0] # Valor de pitch adicionado ao dicionário

        # Procurar o valor de thumbfov, campo de visão (zoom), caso não exista no URL o valor padrão é 90
        thumbfov_start = url_decoded.find("thumbfov=")
        if thumbfov_start != -1:
            thumbfov_start += len("thumbfov=")
            thumbfov_end = url_decoded.find("&", thumbfov_start)
            if thumbfov_end == -1:
                thumbfov_end = len(url_decoded)
            parametros['thumbfov'] = url_decoded[thumbfov_start:thumbfov_end] # Valor de thumbfov adicionada ao dicionário
        else:
            parametros['thumbfov'] = '90'

        # Procurar as coordenadas da localização a partir de @ e até ao final antes de /data
        position_start = url_decoded.find("@") + 1
        position_end = url_decoded.find("/data")
        position_data = url_decoded[position_start:position_end].split(",")

        # Adiciona os valores descobertos ao dicionário e procura outros parâmetros como zoom, heading e tilt
        parametros['latitude'] = position_data[0]   # position_data é um array
        parametros['longitude'] = position_data[1]
        parametros['zoom'] = position_data[2].replace("a", "") if len(position_data) > 2 else "1"
        parametros['heading'] = position_data[3].replace("h", "") if len(position_data) > 3 else "0"
        parametros['tilt'] = position_data[4].replace("t", "") if len(position_data) > 4 else "0"

    # Caso exista URLs malformados ou ausentes aparece uma mensagem no terminal com a exceção detalhada
    except Exception as e:
        print(f"Erro ao extrair parâmetros: {e}")

    # Retorna o dicionário que contem todos os valores extraídos do URL
    return parametros


def descarregar_imagem_street_view(url, output_folder):
    # Obtenção dos parâmetros necessários para obter a imagem
    parametros = extrair_parametros(url)

    # Caso a função não consiga extrair os parâmetros do URL (vazio ou None) uma mensagem de erro é exibida e a execução da função é interrompida
    if not parametros:
        print("Não foi possível extrair os parâmetros do URL.")
        return

    print("Parâmetros extraídos:", parametros)

    # Criação de URL da imagem com os parâmetros retirados anteriormente
    # Tamanho da imagem: 900 largura e 600 altura (pixels)
    image_url = (f"https://streetviewpixels-pa.googleapis.com/v1/thumbnail?"
                 f"panoid={parametros['panoid']}&yaw={parametros['yaw']}&pitch={parametros['pitch']}"
                 f"&thumbfov={parametros['thumbfov']}&w=900&h=600"
                 f"&zoom={parametros['zoom']}&heading={parametros['heading']}&tilt={parametros['tilt']}")

    print("URL da imagem:", image_url)

    # Faz se uma requisição HTTP para obter a imagem do URL criado anteriormente
    response = requests.get(image_url)

    # Código de resposta = 200, indica que a requisição foi bem sucedida. Se o código de resposta não for 200 imprime uma mensagem de erro com o código.
    if response.status_code == 200:
        # Se a pasta não existir, cria
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Listar todos os ficheiros presentes na pasta e extrai os números de índice, padrão é id_indice.jpg
        ficheiros_existentes = os.listdir(output_folder)
        indices_existentes = [int(f.split('_')[1].split('.')[0]) for f in ficheiros_existentes if f.startswith(output_folder.split(os.sep)[-1] + '_')]
        # O próximo índice é calculado com base no maior número,se não existir o índice começa em 1
        proximo_indice = max(indices_existentes) + 1 if indices_existentes else 1
        image_name = f"{output_folder.split(os.sep)[-1]}_{proximo_indice}"

        # O nome novo é construído, id_indice.jpg
        image_path = os.path.join(output_folder, f"{image_name}.jpg")

        with open(image_path, 'wb') as f:   # wb indica que o ficheiro será escrito em binário
            f.write(response.content)

        print(f"Imagem guardada: {image_path}") # Mensagem de suceso é impressa
    else:
        print(f"Erro ao guardar. Status code: {response.status_code}") # Mensagem de erro +e impressa


def main():
    while True:
        url = input("URL: ")
        output_folder = "Fotos"

        escolha = input("\nNúmero da pasta: ")
        output_folder = os.path.join(output_folder, escolha)

        descarregar_imagem_street_view(url, output_folder)

        continuar = input("\nMais fotos? (s/n): ")
        if continuar.lower() != 's':
            break


if __name__ == "__main__":
    main()
