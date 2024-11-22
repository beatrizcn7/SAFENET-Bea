# ----------------- Bibliotecas ---------------
import tensorflow as tf # Principal biblioteca de Machine Learnig e Deep Learning. Cria, treina e implementa modelos de resdes neurais.
from tensorflow.keras import layers, models # API de alto nível que facilita a construção e treino de redes neurais dentro do TensorFlow
from tensorflow.keras.applications import ResNet50 # Modelo pré-treinado utilizado
import pandas as pd # Serve para manipulação e análise de dados
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

# Cria um novo modelo
model = models.Sequential()
model.add(base_model)
model.add(layers.Flatten())
model.add(layers.Dense(11, activation='softmax'))  # 11 classes

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

# Pastas dos ficheiros TFRecords (Treino e Teste)
train_tfrecords = tf.io.gfile.glob('Pasta Final TFRecord - Material + Ano + Estrutura/Treino/*.tfrecord')
val_tfrecords = tf.io.gfile.glob('Pasta Final TFRecord - Material + Ano + Estrutura/Validação/*.tfrecord')

# Carregar os ficheiros TFRecords
train_dataset = load_dataset(train_tfrecords).batch(32).prefetch(tf.data.AUTOTUNE)
val_dataset = load_dataset(val_tfrecords).batch(32).prefetch(tf.data.AUTOTUNE)

# Início da contagem de tempo
start_time = time.time()

# Treinar o modelo
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10
)

# Guardar o modelo treinado
# model.save('Tentativa 1 - Material + Ano + Estrutura (sem aprender).h5')
# print('Guardado o modelo!')

# Carregar o conjunto de teste
test_tfrecords = tf.io.gfile.glob('Pasta Final TFRecord - Material + Ano + Estrutura/Teste/*.tfrecord')
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

# ----------------- Representação dos dados -----------------
# Salvar os resultados do treinamento em um Excel
results = {
    'Epoch': list(range(1, len(history.history['loss']) + 1)),
    'Loss': history.history['loss'],
    'Accuracy': history.history['accuracy'],
    'Val_Loss': history.history['val_loss'],
    'Val_Accuracy': history.history['val_accuracy'],
}

# Criar um DataFrame e salvar como Excel
df_results = pd.DataFrame(results)
df_results.to_excel('Ao longo das épocas.xlsx', index=False)

# Imprimir os resultados do treino
print("Resultados guardados")

# Criar gráficos de Loss e Accuracy
epochs = results['Epoch']
train_loss = results['Loss']
val_loss = results['Val_Loss']
train_accuracy = results['Accuracy']
val_accuracy = results['Val_Accuracy']

# Melhor epoch para loss e accuracy
best_loss_epoch = epochs[val_loss.index(min(val_loss))]
best_loss = min(val_loss)
best_accuracy_epoch = epochs[val_accuracy.index(max(val_accuracy))]
best_accuracy = max(val_accuracy)

# Configurar o tamanho da figura
plt.figure(figsize=(12, 6))

# Gráfico 1: Loss
plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss, label='Treino', marker='o', color='blue')
plt.plot(epochs, val_loss, label='Validação', marker='o', color='red')
plt.axvline(best_loss_epoch, linestyle='--', color='gray', label=f'Melhor Epoch: {best_loss_epoch}')
plt.scatter(best_loss_epoch, best_loss, color='red', label=f'Menor Val Loss: {best_loss:.4f}')
plt.title('Loss ao Longo dos Épocas')
plt.xlabel('Épocas')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Gráfico 2: Accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs, train_accuracy, label='Treino', marker='o', color='blue')
plt.plot(epochs, val_accuracy, label='Validação', marker='o', color='red')
plt.axvline(best_accuracy_epoch, linestyle='--', color='gray', label=f'Melhor Epoch: {best_accuracy_epoch}')
plt.scatter(best_accuracy_epoch, best_accuracy, color='green', label=f'Maior Val Accuracy: {best_accuracy:.4f}')
plt.title('Accuracy ao Longo dos Épocas')
plt.xlabel('Épocas')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Salvar os gráficos como imagem
plt.tight_layout()
plt.savefig('Loss e Accuracy.png')
plt.show()

print("Gráficos de Loss e Accuracy done!'")


# Função para calcular as métricas
def calcular_metricas(y_true, y_pred):
    # Calcular a matriz de confusão
    cm = confusion_matrix(y_true, y_pred)

    # Extrair TP, FP, TN, FN para cada classe
    TP = np.diagonal(cm)  # Verdadeiros positivos
    FP = np.sum(cm, axis=0) - TP  # Falsos positivos
    FN = np.sum(cm, axis=1) - TP  # Falsos negativos
    TN = np.sum(cm) - (TP + FP + FN)  # Verdadeiros negativos

    # Calcular as métricas
    exatidao = (TP + TN) / (TP + TN + FP + FN)
    precisao = TP / (TP + FP)
    recuperacao = TP / (TP + FN)
    f1 = 2 * (precisao * recuperacao) / (precisao + recuperacao)

    # Criar um dicionário com as métricas para cada classe
    metricas = {
        'Classe': np.arange(len(TP)),
        'Exatidão': exatidao,
        'Precisão': precisao,
        'Recuperação': recuperacao,
        'F1': f1
    }

    return metricas

# Previsões no conjunto de teste
y_true = []  # Verdadeiros rótulos
y_pred = []  # Rótulos previstos

for images, labels in test_dataset:
    predictions = model.predict(images)
    y_true.extend(labels.numpy())  # Verdadeiros rótulos
    y_pred.extend(np.argmax(predictions, axis=1))  # Rótulos previstos

# Calcular as métricas
metricas = calcular_metricas(y_true, y_pred)

# Criar um DataFrame com as métricas
df_metricas = pd.DataFrame(metricas)

# Salvar o DataFrame em um novo arquivo Excel
df_metricas.to_excel('Metricas.xlsx', index=False)

# Imprimir que o Excel foi salvo
print("Excel com as métricas guardadas")
