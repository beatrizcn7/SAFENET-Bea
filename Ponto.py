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
url = ("https://www.google.com/maps/@40.4873193,-7.5991856,3a,90y,201.84h,110.8t/data=!3m7!1e1!3m5!1s_bTaq-06Dzh7-faI4fzQXg!2e0!6shttps:%2F%2Fstreetviewpixels-pa.googleapis.com%2Fv1%2Fthumbnail%3Fpanoid%3D_bTaq-06Dzh7-faI4fzQXg%26cb_client%3Dmaps_sv.share%26w%3D900%26h%3D600%26yaw%3D201.83614399567242%26pitch%3D-20.804615842802832%26thumbfov%3D90!7i16384!8i8192?coh=205410&entry=ttu")







# Pasta de saída e nome da imagem
output_folder = "Fotos"
image_name = "114_30.jpg"

# Baixar a imagem
baixar_imagem(url, output_folder, image_name)
