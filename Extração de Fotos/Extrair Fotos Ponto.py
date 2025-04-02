# Beatriz Neves, 9 de agosto de 2024
# Input = Informação no código
# Output = Fotos
# Objetivo = Ajuda a guardar fotos com os URLs de Google Street View, mas agora em pontos

import requests
import os

def baixar_imagem(url, output_folder, image_name):
    try:
        # Usar diretamente a URL fornecida para baixar a imagem
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'image/jpeg',
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        if response.status_code == 200:
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            image_path = os.path.join(output_folder, f"{image_name}.jpg")
            with open(image_path, 'wb') as f:
                f.write(response.content)

            print(f"Imagem salva em: {image_path}")
        else:
            print(f"Erro ao baixar a imagem. Status code: {response.status_code}")

    except requests.exceptions.RequestException as e:
        print(f"Erro durante a requisição HTTP: {e}")

# URL fornecida (modifique conforme necessário)
url = "https://www.google.com/maps/@41.1396599,-8.610322,3a,48.9y,18.98h,105.42t/data=!3m8!1e1!3m6!1sAF1QipOWQF-WuWfiwnjzgl60HxGE-A1U_Z1eQQj2sJv5!2e10!3e11!6s%2F%2Flh5.ggpht.com%2Fp%2FAF1QipOWQF-WuWfiwnjzgl60HxGE-A1U_Z1eQQj2sJv5%3Dw900-h600-k-no-pi-15.415516616999483-ya332.98127208924717-ro0-fo100!7i10240!8i5120?coh=205410&entry=ttu&g_ep=EgoyMDI0MDgyMS4wIKXMDSoASAFQAw%3D%3D"

output_folder = "Fotos"
# nome da foto (modifique conforme necessário)
image_name = "1255_"

baixar_imagem(url, output_folder, image_name)
