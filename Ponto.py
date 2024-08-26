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
url = "https://www.google.com/maps/@41.1881684,-7.5422142,3a,75y,351.47h,97.93t/data=!3m8!1e1!3m6!1sAF1QipMmLr1Now3Gt7fY-81EWmverrCf_PZj9rrLye2S!2e10!3e11!6s%2F%2Flh5.ggpht.com%2Fp%2FAF1QipMmLr1Now3Gt7fY-81EWmverrCf_PZj9rrLye2S%3Dw900-h600-k-no-pi-7.932545542410409-ya351.47260293913433-ro0-fo100!7i6912!8i3456?coh=205410&entry=ttu"


output_folder = "Fotos"
image_name = "1016_3"

baixar_imagem(url, output_folder, image_name)
