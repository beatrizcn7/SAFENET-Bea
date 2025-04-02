# Beatriz Neves, 24 de setembro de 2024
# Input = Pasta Final TFRecord (Treino, Teste e Validação)
# Output = Impressão da loss e accuracy no terminal assim como o tempo, e os diversos ficheiros
# (Accuracy, F1 Score, Loss e Accuracy ao longo das épocas, Matriz de Confusão, Métricas, Precision, Recall, ROC 0 a 20, ROC 22 a 42)
# Objetivo = Treinar um modelo ResNet50 com a informação input, mesmo que o script Tentativa 1, mas sem os ficheiros de output


# ----------------- Bibliotecas ---------------
import tensorflow as tf # Principal biblioteca de Machine Learnig e Deep Learning. Cria, treina e implementa modelos de resdes neurais.
from tensorflow.keras import layers, models # API de alto nível que facilita a construção e treino de redes neurais dentro do TensorFlow
from tensorflow.keras.applications import ResNet50 # Modelo pré-treinado utilizado
import pandas as pd # Serva para manipulação e análise de dados
import matplotlib.pyplot as plt # Para criar gráficos de diferentes tipos
import seaborn as sns # Para criar gráficos mais elaborados
from sklearn.metrics import confusion_matrix, roc_curve, auc # Usada para calcular a matriz de confusão, e a curva ROC e AUC
import numpy as np # Útil para operações matemáticas
import time  # Biblioteca para contar o tempo
from sklearn.preprocessing import label_binarize # Transforma um array de classes numa matriz binária

# --------------------- Modelo ------------------
# Cria uma instância do modelo ResNet50 com pesos pré-treinados do ImageNet
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Congelar as camadas do modelo base
for layer in base_model.layers:
    layer.trainable = False

# Aumentar os dados de treino com rotações, refleção, zoom, contraste e brilho aleatoriamente
data_augmentation = models.Sequential([
    layers.RandomFlip("horizontal_and_vertical"), # verticalmente ou horizontalmente
    layers.RandomRotation(0.2), # até 20%
    layers.RandomZoom(0.2), # até 20%
    layers.RandomContrast(0.2), # até 20%
    layers.RandomBrightness(0.2), # até 20%
], name="data_augmentation")

# Criar um novo modelo
model = models.Sequential([
    layers.Input(shape=(224, 224, 3)), # Definir a forma de entrada
    data_augmentation,  # Adicionar a camada de aumento de dados
    base_model,
    layers.Flatten(),
    layers.Dense(1024, activation='relu'), # Camada conectada com 1024 neurónios e a função ativação relu
    layers.Dropout(0.5), # dropout com uma taxa de 50%
    layers.Dense(43, activation='softmax')  # As labels vão da 0 à 42 (43 combinações possíveis)
])

# Compilar o modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Resumir o modelo
# model.summary()

# Função para carregar TFRecords
def parse_tfrecord(example):
    # Defina o formato do TFRecord
    feature_description = {
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        'image/object/class/label': tf.io.FixedLenFeature([], tf.int64),
    }
    return tf.io.parse_single_example(example, feature_description)

def load_dataset(file_paths):
    # Cria um dataset a partir de múltiplos ficheiros TFRecord
    raw_dataset = tf.data.TFRecordDataset(file_paths)
    return raw_dataset.map(parse_tfrecord).map(
        lambda x: (
            tf.image.resize(tf.image.decode_jpeg(x['image/encoded'], channels=3), [224, 224]),
            x['image/object/class/label']
        )
    )

# Pastas dos ficheiros TFRecords (Treino e Validação)
train_tfrecords = tf.io.gfile.glob('Pasta Final TFRecord/Treino/*.tfrecord')
val_tfrecords = tf.io.gfile.glob('Pasta Final TFRecord/Validação/*.tfrecord')

# Carregar os datasets
train_dataset = load_dataset(train_tfrecords).batch(32).prefetch(tf.data.AUTOTUNE)
val_dataset = load_dataset(val_tfrecords).batch(32).prefetch(tf.data.AUTOTUNE)

# Início da contagem de tempo
start_time = time.time()

# Interrompe o treino do modelo de forma antecipada se o desempenho no conjunto de validação não melhorar
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Dicionário para armazenar métricas de treino e validação ao longo das épocas
history_metrics = {
    "epoch": [],
    "loss": [],
    "accuracy": [],
    "val_loss": [],
    "val_accuracy": []
}

# Treinar o modelo
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10,
    callbacks=[early_stopping] # Pode parar o treino se o desempenho nos dados de validação não melhorar
)

# Percorre cada época e guarda os valores de loss e accuracy para treino e validação
for epoch in range(len(history.history['loss'])):
    history_metrics['epoch'].append(epoch + 1)
    history_metrics['loss'].append(history.history['loss'][epoch])
    history_metrics['accuracy'].append(history.history['accuracy'][epoch])
    history_metrics['val_loss'].append(history.history['val_loss'][epoch])
    history_metrics['val_accuracy'].append(history.history['val_accuracy'][epoch])

# Ativa o treino das últimas 30 camadas
for layer in base_model.layers[-30:]:
    layer.trainable = True

# Compilar o modelo usando o otimizador Adam com uma taxa de aprendizagem
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Inicia o treino do modelo por 10 épocas e implementa um callback
fine_tune_history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10,
    callbacks=[early_stopping]
)

# Percorre cada época (a partir da época 11) e guarda os valores de loss e accuracy para treino e validação
for epoch in range(len(fine_tune_history.history['loss'])):
    history_metrics['epoch'].append(epoch + 11)  # Continuar a numeração das épocas
    history_metrics['loss'].append(fine_tune_history.history['loss'][epoch])
    history_metrics['accuracy'].append(fine_tune_history.history['accuracy'][epoch])
    history_metrics['val_loss'].append(fine_tune_history.history['val_loss'][epoch])
    history_metrics['val_accuracy'].append(fine_tune_history.history['val_accuracy'][epoch])

# Carregar o conjunto de teste
test_tfrecords = tf.io.gfile.glob('Pasta Final TFRecord/Teste/*.tfrecord')
test_dataset = load_dataset(test_tfrecords).batch(32).prefetch(tf.data.AUTOTUNE)

# Avaliar o modelo no conjunto de teste
test_loss, test_accuracy = model.evaluate(test_dataset)

end_time = time.time()  # Fim
total_time = (end_time - start_time) / 60  # Converter para minutos
# Imprimir o tempo total
print(f"Tempo total: {total_time:.2f} minutos")

# Imprime os resultados da perda e da exatidão dos dados teste
print(f'Teste Loss: {test_loss}')
print(f'Teste Accuracy: {test_accuracy}')

# ------------- Representação de dados ----------
# Guardar o dataframe de resultados num arquivo Excel
df_metrics = pd.DataFrame(history_metrics)
df_metrics.to_excel('Resultados.xlsx', index=False)

print('Excel criado')

plt.figure(figsize=(12, 5))

# Identificar a melhor epoch em Accuracy
best_val_accuracy = max(history_metrics['val_accuracy'])
best_epoch_accuracy = history_metrics['epoch'][history_metrics['val_accuracy'].index(best_val_accuracy)]

# Identificar a melhor epoch em Loss
best_val_loss = min(history_metrics['val_loss'])
best_epoch_loss = history_metrics['epoch'][history_metrics['val_loss'].index(best_val_loss)]

# Gráfico 1: Accuracy ao longo das épocas
plt.subplot(1, 2, 2)
plt.plot(history_metrics['epoch'], history_metrics['accuracy'], label='Accuracy de Treino')
plt.plot(history_metrics['epoch'], history_metrics['val_accuracy'], label='Accuracy de Validação')
plt.axvline(x=best_epoch_accuracy, color='r', linestyle='--', label=f'Melhor Epoch {best_epoch_accuracy}')
plt.title('Accuracy ao longo dos Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Gráfico 2: Loss ao longo das épocas
plt.subplot(1, 2, 1)
plt.plot(history_metrics['epoch'], history_metrics['loss'], label='Loss de Treino')
plt.plot(history_metrics['epoch'], history_metrics['val_loss'], label='Loss de Validação')
plt.axvline(x=best_epoch_loss, color='r', linestyle='--', label=f'Melhor Epoch {best_epoch_loss}')
plt.title('Loss ao longo dos Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()


# Ajustar layout e mostrar os gráficos
plt.tight_layout()
plt.savefig('Loss e Accuracy.png')  # Guardar os gráficos numa imagem
plt.show()

print('Gráficos de Accuracy e Loss criados')

y_true = [] # armazena as labels verdadeiras de cada batch dos dados teste
y_pred = [] # armazena as previsões feitas pelo modelo
y_pred_probs = [] # armazena as probabilidades previstas para cada classe

# Recolher as labels verdadeiras e as previsões
for images, labels in test_dataset:
    preds = model.predict(images)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))
    y_pred_probs.extend(preds)

# Colocar binário as classes
y_true_binarized = label_binarize(y_true, classes=range(43))

# Calcular as classes com casos de teste
classes_with_samples = np.unique(y_true)  # Descobrir automaticamente as classes com casos de teste

# Definir o layout dos gráficos para cada intervalo de classes
def plot_roc_for_classes(classes, n_rows, n_cols, file_name):
    plt.figure(figsize=(n_cols * 5, n_rows * 4))
    for idx, cls in enumerate(classes):
        if cls in classes_with_samples:  # Verifica se há casos de teste para essa classe
            fpr, tpr, _ = roc_curve(y_true_binarized[:, cls], np.array(y_pred_probs)[:, cls])
            roc_auc = auc(fpr, tpr)
            plt.subplot(n_rows, n_cols, idx + 1)
            plt.plot(fpr, tpr, color='blue', label=f'Classe {cls} (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Falso Positivo')
            plt.ylabel('Verdadeiro Positivo')
            plt.title(f'ROC Classe {cls}')
            plt.legend(loc="lower right")
        else:
            plt.subplot(n_rows, n_cols, idx + 1)
            plt.text(0.5, 0.5, 'Sem dados', horizontalalignment='center', verticalalignment='center', fontsize=12)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Falso Positivo')
            plt.ylabel('Verdadeiro Positivo')
            plt.title(f'ROC Classe {cls}')

    plt.tight_layout()
    plt.savefig(file_name)
    plt.show()

# Gerar gráfico para classes 0 a 20 (7 linhas e 3 colunas)
plot_roc_for_classes(range(0, 21), 7, 3, 'ROC 0 a 20.png')

# Gerar gráfico para classes 21 a 42 (11 linhas e 2 colunas)
plot_roc_for_classes(range(21, 43), 11, 2, 'ROC 21 a 42.png')

print('Gráficos ROC e AUC feitos.')

# Calcular a matriz de confusão
cm = confusion_matrix(y_true, y_pred, labels=range(43))

# Guardar a matriz de confusão num ficheiro Excel
df_cm = pd.DataFrame(cm, index=[f'Verdadeiro {i}' for i in range(43)], columns=[f'Previsto {i}' for i in range(43)])
df_cm.to_excel('Matriz de Confusão.xlsx', index=True)

# Gráfico da matriz de confusão
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(43), yticklabels=range(43))
plt.title('Confusion Matrix')
plt.xlabel('Previsto Label')
plt.ylabel('Verdadeiro Label')
plt.savefig('Matriz de Confusão.png')
plt.show()

print('Matriz de confusão criada')

# Inicializar as listas para armazenar as métricas (exatidão, precisão, recuperação e F1)
accuracies, precisions, recalls, f1_scores = [], [], [], []

# Calcular as métricas para cada label
for i in range(len(cm)):
    TP = cm[i, i]
    FP = np.sum(cm[:, i]) - TP
    FN = np.sum(cm[i, :]) - TP
    TN = np.sum(cm) - (TP + FP + FN)

    # Exatidão
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    accuracies.append(accuracy)

    # Precisão
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    precisions.append(precision)

    # Recuperação
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    recalls.append(recall)

    # F1
    f1 = 2 * ((precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0
    f1_scores.append(f1)

# Criar DataFrame com as métricas
df_metrics_final = pd.DataFrame({
    'Label': range(43),
    'Accuracy': accuracies,
    'Precision': precisions,
    'Recall': recalls,
    'F1 Score': f1_scores
})

# Guardar as métricas num Excel
df_metrics_final.to_excel("Métricas.xlsx", index=False)

print('Métricas calculadas')

# Fazer os gráficos para cada métrica individual e guardar separadamente
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
file_names = ['Accuracy.png', 'Precision.png', 'Recall.png', 'F1 Score.png']

for metric, file_name in zip(metrics, file_names):
    plt.figure(figsize=(8, 5))
    plt.plot(df_metrics_final['Label'], df_metrics_final[metric], marker='o')
    plt.title(f'{metric} por Label')
    plt.xlabel('Label')
    plt.ylabel(metric)
    plt.xticks(df_metrics_final['Label'])  # Mostrar todas as labels no eixo x
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(file_name) # Guardar gráfico como imagem
    plt.show()
    print(f'Gráfico {metric} salvo como {file_name}')
