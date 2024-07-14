# -*- coding: utf-8 -*-

import requests
import os
from urllib.parse import unquote


def extrair_parametros(url):
    parametros = {}
    url_decoded = unquote(url)

    try:
        panoid_start = url_decoded.find("panoid=") + len("panoid=")
        panoid_end = url_decoded.find("&", panoid_start)
        parametros['panoid'] = url_decoded[panoid_start:panoid_end]

        yaw_start = url_decoded.find("yaw=") + len("yaw=")
        yaw_end = url_decoded.find("&", yaw_start)
        parametros['yaw'] = url_decoded[yaw_start:yaw_end]

        pitch_start = url_decoded.find("pitch=") + len("pitch=")
        pitch_end = url_decoded.find("&", pitch_start)
        parametros['pitch'] = url_decoded[pitch_start:pitch_end]

        thumbfov_start = url_decoded.find("thumbfov=") + len("thumbfov=")
        thumbfov_end = url_decoded.find("&", thumbfov_start)
        if thumbfov_end == -1:  # Caso não encontre '&', pegar até o final da string
            thumbfov_end = len(url_decoded)
        parametros['thumbfov'] = url_decoded[thumbfov_start:thumbfov_end].split('!')[0]  # Corrigir para capturar apenas o valor numérico

        position_start = url_decoded.find("@") + 1
        position_end = url_decoded.find("/data")
        position_data = url_decoded[position_start:position_end].split(",")

        parametros['zoom'] = position_data[2].replace("a", "") if len(position_data) > 2 else "1"
        parametros['heading'] = position_data[3].replace("h", "") if len(position_data) > 3 else "0"
        parametros['tilt'] = position_data[4].replace("t", "") if len(position_data) > 4 else "0"

    except Exception as e:
        print(f"Erro ao extrair parâmetros: {e}")

    return parametros


def baixar_imagem_street_view(url, output_folder, image_name):
    parametros = extrair_parametros(url)

    if not parametros:
        print("Não foi possível extrair os parâmetros da URL.")
        return

    print("Parâmetros extraídos:", parametros)

    image_url = (f"https://streetviewpixels-pa.googleapis.com/v1/thumbnail?"
                 f"panoid={parametros['panoid']}&yaw={parametros['yaw']}&pitch={parametros['pitch']}"
                 f"&thumbfov={parametros['thumbfov']}&w=900&h=600"
                 f"&zoom={parametros['zoom']}&heading={parametros['heading']}&tilt={parametros['tilt']}")

    print("URL da imagem:", image_url)

    response = requests.get(image_url)

    if response.status_code == 200:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)


        image_path = os.path.join(output_folder, f"{image_name}.jpg")
        with open(image_path, 'wb') as f:
            f.write(response.content)

        print(f"Imagem salva em: {image_path}")
    else:
        print(f"Erro ao baixar a imagem. Status code: {response.status_code}")


url = ("https://www.google.com/maps/@40.4869209,-7.5988854,3a,75y,45.88h,105.04t/data=!3m7!1e1!3m5!1sqBNH0zGbDWK4-NNvY3SFqQ!2e0!6shttps:%2F%2Fstreetviewpixels-pa.googleapis.com%2Fv1%2Fthumbnail%3Fpanoid%3DqBNH0zGbDWK4-NNvY3SFqQ%26cb_client%3Dmaps_sv.share%26w%3D900%26h%3D600%26yaw%3D45.8833110148118%26pitch%3D-15.036784878669437%26thumbfov%3D90!7i16384!8i8192?coh=205410&entry=ttu")






output_folder = "Fotos"
image_name = ("114_4")

baixar_imagem_street_view(url, output_folder, image_name)