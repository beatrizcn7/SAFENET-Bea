# ----------------- Bibliotecas ---------------
# Importar a principal biblioteca de Machine Learning e Deep Learning.
# Criar, treinar e implementar modelos de redes neurais.
import tensorflow as tf
# Importar a API de alto nível que facilita a construção e treino de redes neurais dentro do TensorFlow
from tensorflow.keras import layers, models
# Importar o modelo pré-treinado ResNet50
from tensorflow.keras.applications import ResNet50
# Utilizar para manipulação e análse de dados
import pandas as pd
# Para criar gráficos de diferentes tipos
import matplotlib.pyplot as plt
# Para criar gráficos mais elaborados
import seaborn as sns
# Utilizar para calcular a matriz de confusão, e a curva ROC e AUC
from sklearn.metrics import confusion_matrix, roc_curve, auc
# Utilizar para operações matemáticas
import numpy as np
# Importar a biblioteca para contar o tempo
import time
# Transformar um array de classes numa matriz binária
from sklearn.preprocessing import label_binarize


# --------------------- Modelo ------------------
# Carregar o modelo salvo para a propriedade "Material"
base_model = tf.keras.models.load_model('Tentativa 1 - Material.h5')

# Remover a última camada do modelo base
base_model = models.Model(inputs=base_model.input, outputs=base_model.layers[-2].output)

# Congelar as camadas do modelo base
for layer in base_model.layers:
    layer.trainable = False

# Criar um novo modelo na parte superior
inputs = layers.Input(shape=(224, 224, 3))  # Definir a entrada do modelo com tamanho 224x224 e 3 canais de cor
x = base_model(inputs, training=False)  # Passar a entrada pela base do modelo (sem treino das camadas base)

# Criar uma camada densa com 5 classes (saída da classificação) e função de ativação softmax
outputs = layers.Dense(6, activation='softmax')(x)

# Criar o modelo final
model = models.Model(inputs, outputs)  # Definir o modelo com as entradas e saídas especificadas

# Compilar o modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Função para carregar e processar os dados dos ficheiros TFRecord
def parse_tfrecord(example):
    # Definir o formato dos dados no TFRecord
    feature_description = {
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        'image/object/class/label': tf.io.FixedLenFeature([], tf.int64),
    }
    return tf.io.parse_single_example(example, feature_description)

# Função para carregar o dataset a partir de vários ficheiros TFRecord
def load_dataset(file_paths):
    # Criar um dataset a partir de múltiplos arquivos TFRecord
    raw_dataset = tf.data.TFRecordDataset(file_paths)
    return raw_dataset.map(parse_tfrecord).map(
        lambda x: (
            tf.image.resize(tf.image.decode_jpeg(x['image/encoded'], channels=3), [224, 224]),
            x['image/object/class/label']
        )
    )

# Pastas dos ficheiros TFRecords (Treino e Teste)
train_tfrecords = tf.io.gfile.glob('Pasta Final TFRecord - Material + Estrutura/Treino/*.tfrecord')
val_tfrecords = tf.io.gfile.glob('Pasta Final TFRecord - Material + Estrutura/Validação/*.tfrecord')

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
model.save('Tentativa 1 - Material + Estrutura (Do Zero).h5')
print('Guardado o modelo!')

# Carregar o conjunto de teste
test_tfrecords = tf.io.gfile.glob('Pasta Final TFRecord - Material + Estrutura/Teste/*.tfrecord')
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

# EXCEL DA LOSS E ACCURACY AO LONGO DAS ÉPOCAS

# Criar um dataframe com os resultados de cada época
results_df = pd.DataFrame({
    'Epoch': range(1, len(history.history['accuracy']) + 1),
    'Loss': history.history['loss'],
    'Accuracy': history.history['accuracy'],
    'Val Loss': history.history['val_loss'],
    'Val Accuracy': history.history['val_accuracy']
})

# Guardar o dataframe de resultados num ficheiro Excel
results_df.to_excel('Ao longo das Épocas.xlsx', index=False)

print('Excel da loss e accuracy ao longo das épocas criado')

# GRÁFICO DE LOSS E ACCURACY AO LONGO DAS ÉPOCAS

# Fazer os gráficos de accuracy e loss
epochs = range(1, len(history.history['accuracy']) + 1)

# Identificar a melhor epoch
best_epoch_acc = epochs[history.history['val_accuracy'].index(max(history.history['val_accuracy']))]
best_epoch_loss = epochs[history.history['val_loss'].index(min(history.history['val_loss']))]

plt.figure(figsize=(12, 5))

# Gráfico 1: Accuracy ao longo das épocas
plt.subplot(1, 2, 1)
plt.plot(epochs, history.history['accuracy'], label='Accuracy de Treino', marker='o')
plt.plot(epochs, history.history['val_accuracy'], label='Accuracy de Validação', marker='o')
plt.axvline(x=best_epoch_acc, color='r', linestyle='--', label=f'Melhor Epoch {best_epoch_acc}')
plt.title('Accuracy ao longo dos Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Gráfico 2: Loss ao longo das épocas
plt.subplot(1, 2, 2)
plt.plot(epochs, history.history['loss'], label='Loss de Treino', marker='o')
plt.plot(epochs, history.history['val_loss'], label='Loss de Validação', marker='o')
plt.axvline(x=best_epoch_loss, color='r', linestyle='--', label=f'Melhor Epoch {best_epoch_loss}')
plt.title('Loss ao longo dos Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Ajustar layout e mostrar os gráficos
plt.tight_layout()
plt.savefig('Loss e Accuracy.png')  # Guardar os gráficos numa imagem
plt.show()

print('Gráficos da loss e accuracy ao longo das épocas criado')

# EXCEL DAS MÉTRICAS (EXATIDÃO, PRECISÃO, RECUPERAÇÃO E F1)

y_true = [] # armazenar as classess verdadeiras de cada batch dos dados teste
y_pred = [] # armazenar as previsões feitas pelo modelo
y_pred_probs = [] # armazenar as probabilidades previstas para cada classe

# Recolher as classes verdadeiras e as previsões
for images, labels in test_dataset:
    preds = model.predict(images)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))
    y_pred_probs.extend(preds)

# Colocar binário as classes
y_true_binarized = label_binarize(y_true, classes=range(6))

# Calcular as classes com casos de teste
classes_with_samples = np.unique(y_true)  # Descobrir automaticamente as classes com casos de teste

# Calcular a matriz de confusão
cm = confusion_matrix(y_true, y_pred, labels=range(6))

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

# Criar um dataframe com as métricas
metrics_df = pd.DataFrame({
    'Classe': range(6),
    'Accuracy': accuracies,
    'Precision': precisions,
    'Recall': recalls,
    'F1 Score': f1_scores
})

# Salvar o dataframe das métricas num ficheiro Excel
metrics_df.to_excel('Métricas.xlsx', index=False)

print('Excel das Métricas criados')

# GRÁFICO DAS MÉTRICAS

# Lista de métricas a serem visualizadas
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
# Nomes dos ficheiros onde os gráficos serão salvos
file_names = ['Accuracy.png', 'Precision.png', 'Recall.png', 'F1 Score.png']

for metric, file_name in zip(metrics, file_names):
    plt.figure(figsize=(8, 5))
    plt.plot(metrics_df['Classe'], metrics_df[metric], marker='o')
    plt.title(f'{metric} por Classe')
    plt.xlabel('Classe')
    plt.ylabel(metric)
    plt.xticks(metrics_df['Classe'])  # Mostrar todas as labels no eixo x
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(file_name)  # Guardar gráfico como imagem
    plt.show()

print('Gráficos individuais criados.')

# FOTO DA MATRIZ DE CONFUSÃO

# Criar o gráfico da matriz de confusão
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(6), yticklabels=range(6))
plt.title('Matriz de Confusão')
plt.xlabel('Classe Prevista')
plt.ylabel('Classe Verdadeira')
plt.savefig('Matriz de Confusão.png')  # Guardar a matriz de confusão como imagem
plt.show()

print('Foto da Matriz de confusão criada')

# EXCEL DA MATRIZ DE CONFUSÃO

# Criar um dataframe a partir da matriz de confusão
cm_df = pd.DataFrame(cm, index=[f'Verdadeiro {i}' for i in range(6)], columns=[f'Previsto {i}' for i in range(6)])

# Salvar o dataframe num ficheiro Excel
cm_df.to_excel('Matriz de Confusão.xlsx', index=True)

print('Excel da Matriz de Confusão criado')

# GRÁFICO ROC E AUC

# Função para fazer o gráfico de ROC para as classes
def plot_roc_for_three_classes(classes, file_name):
    plt.figure(figsize=(20, 5))

    for idx, cls in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_true_binarized[:, cls], np.array(y_pred_probs)[:, cls])
        roc_auc = auc(fpr, tpr)

        plt.subplot(1, 6, idx + 1)
        plt.plot(fpr, tpr, color='blue', label=f'Classe {cls} (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Falso Positivo')
        plt.ylabel('Verdadeiro Positivo')
        plt.title(f'ROC Classe {cls}')
        plt.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(file_name)
    plt.show()

# Gerar o gráfico para as 6 classes
plot_roc_for_three_classes(range(6), 'Material + Estrutura/ResNet/Tentativa 1/Do Zero/ROC.png')

print('Gráficos ROC e AUC criados.')
