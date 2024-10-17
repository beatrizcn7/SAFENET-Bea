import pandas as pd

# Inicializar dicionários para armazenar os dados
data = {
    "Label": [],
    "Treino": [],
    "Validação": [],
    "Teste": []
}

# Função para ler o arquivo e preencher os dados
def read_labels(file_path):
    current_set = None  # Variável para acompanhar se estamos em treino, validação ou teste
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith("Pasta:"):
                # Atualiza o conjunto atual baseado na linha
                current_set = line.split(": ")[1]  # Pega o nome da pasta (Treino, Validação, Teste)
            elif line.startswith("Label"):
                label, count = line.split(": ")
                label_number = int(label.split()[1])  # Pega apenas o número da label
                count_value = int(count)  # Converte o valor para inteiro

                # Adiciona os dados ao dicionário apropriado
                if current_set == "Treino":
                    data["Treino"].append(count_value)
                elif current_set == "Validação":
                    data["Validação"].append(count_value)
                elif current_set == "Teste":
                    data["Teste"].append(count_value)

                # Certifica-se de que todas as labels estão registradas
                while len(data["Label"]) <= label_number:
                    data["Label"].append(label_number)

# Ler o arquivo Labels.txt
read_labels("Labels.txt")

# Completar listas com 0 para labels não encontradas
max_label = max(data["Label"])
for label in range(max_label + 1):
    if label not in data["Label"]:
        data["Label"].append(label)
        data["Treino"].append(0)
        data["Validação"].append(0)
        data["Teste"].append(0)
    else:
        # Se já existe, preenche com 0 caso não tenha valor
        if len(data["Treino"]) < len(data["Label"]):
            data["Treino"].append(0)
        if len(data["Validação"]) < len(data["Label"]):
            data["Validação"].append(0)
        if len(data["Teste"]) < len(data["Label"]):
            data["Teste"].append(0)

# Criar um DataFrame do pandas
df = pd.DataFrame(data)

# Salvar o DataFrame em um arquivo Excel
df.to_excel("contagem_labels.xlsx", index=False)

print("Arquivo Excel 'contagem_labels.xlsx' criado com sucesso!")
