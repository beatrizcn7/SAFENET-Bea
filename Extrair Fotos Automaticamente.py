import math

# Função para calcular o "heading" entre dois pontos (latitude, longitude)
def calcular_heading(lat1, lng1, lat2, lng2):
    """
    Calcula o heading entre dois pontos geográficos em graus.
    :param lat1: Latitude do ponto 1
    :param lng1: Longitude do ponto 1
    :param lat2: Latitude do ponto 2
    :param lng2: Longitude do ponto 2
    :return: Heading em graus
    """
    d_lng = math.radians(lng2 - lng1)
    lat1 = math.radians(lat1)
    lat2 = math.radians(lat2)

    y = math.sin(d_lng) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(d_lng)
    heading = math.degrees(math.atan2(y, x))
    return (heading + 360) % 360  # Garante que o heading esteja entre 0 e 360


# Função para calcular a direção perpendicular
def calcular_heading_perpendicular(heading):
    """
    Calcula o heading perpendicular a partir de um heading dado.
    :param heading: Heading inicial em graus
    :return: Heading perpendicular em graus
    """
    # Adiciona 90 graus para o sentido horário ou subtrai 90 graus para o sentido anti-horário
    return (heading + 90) % 360  # Acrescenta 90 graus para obter a perpendicular


# Função para calcular o ponto deslocado na direção perpendicular
def deslocar_ponto(lat, lng, heading, distancia):
    """
    Desloca o ponto em uma direção especificada pelo heading e a distância.
    :param lat: Latitude inicial
    :param lng: Longitude inicial
    :param heading: Direção em graus (heading)
    :param distancia: Distância em metros para deslocar o ponto
    :return: Nova latitude e longitude
    """
    # Raio da Terra em metros
    R = 6371000

    # Calcular o deslocamento em lat e lng
    deslocamento_lat = distancia * math.cos(math.radians(heading)) / R
    deslocamento_lng = distancia * math.sin(math.radians(heading)) / (R * math.cos(math.radians(lat)))

    # Calcular nova posição
    nova_lat = lat + math.degrees(deslocamento_lat)
    nova_lng = lng + math.degrees(deslocamento_lng)

    return nova_lat, nova_lng


def main():
    # Entrada de dados - Coordenadas do ponto inicial da ponte
    lat_ponte = float(input("Insira a latitude do ponto inicial da ponte: ").replace(",", "."))
    lng_ponte = float(input("Insira a longitude do ponto inicial da ponte: ").replace(",", "."))

    # Entrada para a distância a ser percorrida na direção da ponte
    distancia_para_frente = float(input("Insira a distância para andar na direção da ponte (em metros): ").replace(",", "."))

    # Entrada para a altura do pilar (distância perpendicular)
    altura_pilar = float(input("Insira a altura do pilar mais alto (em metros): ").replace(",", "."))

    # Coordenada próxima da ponte para determinar o heading
    lat2 = lat_ponte + 0.0001  # Um pequeno incremento de latitude para estimar direção
    lng2 = lng_ponte + 0.0001  # Um pequeno incremento de longitude para estimar direção

    # Calcular o heading entre o ponto da ponte e o ponto próximo
    heading = calcular_heading(lat_ponte, lng_ponte, lat2, lng2)

    # Deslocar o ponto para frente (meia distância)
    ponto_frente_lat, ponto_frente_lng = deslocar_ponto(lat_ponte, lng_ponte, heading, distancia_para_frente / 2)

    # Calcular o heading perpendicular (sentido horário)
    heading_perpendicular = calcular_heading_perpendicular(heading)

    # Deslocar o ponto na perpendicular pela altura do pilar
    nova_lat, nova_lng = deslocar_ponto(ponto_frente_lat, ponto_frente_lng, heading_perpendicular, altura_pilar)

    print(f"Novo ponto final: Latitude: {nova_lat}, Longitude: {nova_lng}")


if __name__ == "__main__":
    main()
