import requests
import os
from urllib.parse import unquote

def extrair_url_imagem(url):
    try:
        # Decodificar a URL
        url_decoded = unquote(url)

        # Procurar a URL da imagem dentro da URL decodificada
        start = url_decoded.find("https://lh5.googleusercontent.com/")
        if start == -1:
            print("URL da imagem não encontrada no link fornecido.")
            return None

        # A URL da imagem pode terminar no próximo '!' ou no final da string
        end = url_decoded.find("!", start)
        if end == -1:
            end = len(url_decoded)

        # Extrair a URL da imagem
        image_url = url_decoded[start:end]
        return image_url

    except Exception as e:
        print(f"Erro ao extrair a URL da imagem: {e}")
        return None

def baixar_imagem(url, output_folder, image_name):
    try:
        # Extrair a URL da imagem
        image_url = extrair_url_imagem(url)

        if not image_url:
            print("Não foi possível extrair a URL da imagem.")
            return

        print("URL da imagem:", image_url)

        # Fazer a requisição para baixar a imagem
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'image/jpeg',
        }
        response = requests.get(image_url, headers=headers, timeout=10)
        response.raise_for_status()

        if response.status_code == 200:
            # Criar a pasta de saída se não existir
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            # Salvar a imagem
            image_path = os.path.join(output_folder, f"{image_name}.jpg")
            with open(image_path, 'wb') as f:
                f.write(response.content)

            print(f"Imagem salva em: {image_path}")
        else:
            print(f"Erro ao baixar a imagem. Status code: {response.status_code}")

    except requests.exceptions.RequestException as e:
        print(f"Erro durante a requisição HTTP: {e}")

# URL fornecida
url = ("https://www.google.com/maps/@40.4016477,-7.3965799,3a,49y,137.06h,93.56t/data=!3m8!1e1!3m6!1sAF1QipPz5lW89V1ZaUcztRXMnCv39_nf-BaidedqMISH!2e10!3e11!6shttps:%2F%2Flh5.googleusercontent.com%2Fp%2FAF1QipPz5lW89V1ZaUcztRXMnCv39_nf-BaidedqMISH%3Dw900-h600-k-no-pi-3.556674958697016-ya141.05970453745408-ro0-fo90!7i8192!8i4096?coh=205410&entry=ttu")




# Pasta de saída e nome da imagem
output_folder = "Fotos"
image_name = "53_2"

# Baixar a imagem
baixar_imagem(url, output_folder, image_name)
