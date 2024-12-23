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
# Carregar o modelo salvo para a propriedade "Material"
model = tf.keras.models.load_model('Tentativa 1 - Material + Ano (do zero).h5')

model.pop()  # Remover a última camada de saída (5 classes)
model.add(layers.Dense(11, activation='softmax'))  #  11 classes

# Congelar as camadas do modelo base, menos a última camada adicionada
for layer in model.layers[:-20]:
    layer.trainable = False

# Compilar o modelo novamente com a nova configuração
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Função para carregar TFRecords
def parse_tfrecord(example):
    # Definir o formato do TFRecord
    feature_description = {
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        'image/object/class/label': tf.io.FixedLenFeature([], tf.int64),
    }
    return tf.io.parse_single_example(example, feature_description)

def load_dataset(file_paths):
    # Criar dataset a partir dos ficheiros TFRecord
    raw_dataset = tf.data.TFRecordDataset(file_paths)
    return raw_dataset.map(parse_tfrecord).map(
        lambda x: (
            tf.image.resize(tf.image.decode_jpeg(x['image/encoded'], channels=3), [224, 224]),
            x['image/object/class/label']
        )
    )

# Pastas dos ficheiros TFRecords (Treino, Validação e Teste)
train_tfrecords = tf.io.gfile.glob('Pasta Final TFRecord - Material + Ano + Estrutura/Treino/*.tfrecord')
val_tfrecords = tf.io.gfile.glob('Pasta Final TFRecord - Material + Ano + Estrutura/Validação/*.tfrecord')

# Carregar os datasets
train_dataset = load_dataset(train_tfrecords).batch(32).prefetch(tf.data.AUTOTUNE)
val_dataset = load_dataset(val_tfrecords).batch(32).prefetch(tf.data.AUTOTUNE)

# Início da contagem de tempo
start_time = time.time()

# Treinar o modelo com os novos dados (Material e Ano)
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10
)

# Guardar o modelo treinado
model.save('Tentativa 1- Material + Ano + Estrutura com Modelo Tentativa 1 - Material + Ano (do zero).h5')
print('Guardado o modelo!')

test_tfrecords = tf.io.gfile.glob('Pasta Final TFRecord - Material + Ano + Estrutura/Teste/*.tfrecord')
test_dataset = load_dataset(test_tfrecords).batch(32).prefetch(tf.data.AUTOTUNE)

# Avaliar o modelo no conjunto de teste
test_loss, test_accuracy = model.evaluate(test_dataset)

# Fim da contagem de tempo
end_time = time.time()
total_time = (end_time - start_time) / 60

# Imprimir o tempo total
print(f"Tempo total: {total_time:.2f} minutos")

# Imprime os resultados da perda e da exatidão dos dados teste
print(f'Teste Loss: {test_loss}')
print(f'Teste Accuracy: {test_accuracy}')

# ----------------- Representação dos dados -----------------

# Criar um dataframe com os resultados de cada época
results_df = pd.DataFrame({
    'Epoch': range(1, len(history.history['accuracy']) + 1),
    'Loss': history.history['loss'],
    'Accuracy': history.history['accuracy'],
    'Val Loss': history.history['val_loss'],
    'Val Accuracy': history.history['val_accuracy']
})

# Guardar o dataframe de resultados num arquivo Excel
results_df.to_excel('Ao longo das Épocas.xlsx', index=False)

print('Excel criado')

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

print('Gráficos de Accuracy e Loss criados')

y_true = [] # armazena as classess verdadeiras de cada batch dos dados teste
y_pred = [] # armazena as previsões feitas pelo modelo
y_pred_probs = [] # armazena as probabilidades previstas para cada classe

# Recolher as classes verdadeiras e as previsões
for images, labels in test_dataset:
    preds = model.predict(images)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))
    y_pred_probs.extend(preds)

# Colocar binário as classes
y_true_binarized = label_binarize(y_true, classes=range(11))


# Calcular a matriz de confusão
cm = confusion_matrix(y_true, y_pred, labels=range(11))

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
    'Classe': range(11),
    'Accuracy': accuracies,
    'Precision': precisions,
    'Recall': recalls,
    'F1 Score': f1_scores
})

# Salvar o dataframe das métricas num arquivo Excel
metrics_df.to_excel('Métricas.xlsx', index=False)

print('Métricas calculadas')

# Fazer os gráficos para cada métrica individual e guardar separadamente
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
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

# Fazer o gráfico da matriz de confusão
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(11), yticklabels=range(11))
plt.title('Matriz de Confusão')
plt.xlabel('Classe Prevista')
plt.ylabel('Classe Verdadeira')
plt.savefig('Matriz de Confusão.png')  # Guardar a matriz de confusão como imagem
plt.show()

print('Matriz de confusão criada')

# Criar um dataframe a partir da matriz de confusão
cm_df = pd.DataFrame(cm, index=[f'Verdadeiro {i}' for i in range(11)], columns=[f'Previsto {i}' for i in range(11)])

# Salvar o dataframe num arquivo Excel
cm_df.to_excel('Matriz de Confusão.xlsx', index=True)

print('Matriz de confusão guardado no Excel')

# Função para fazer o gráfico de ROC para 11 classes
def plot_roc_for_eleven_classes(classes, file_name):
    plt.figure(figsize=(20, 10))  # Ajustar tamanho para 2 linhas

    for idx, cls in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_true_binarized[:, cls], np.array(y_pred_probs)[:, cls])
        roc_auc = auc(fpr, tpr)

        # Organizar em 2 linhas: 6 gráficos na primeira linha, 5 na segunda
        plt.subplot(2, 6, idx + 1)  # Configurar subplot para 2 linhas e 6 colunas (última posição ficará vazia)
        plt.plot(fpr, tpr, color='blue', label=f'Classe {cls} (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Falso Positivo')
        plt.ylabel('Verdadeiro Positivo')
        plt.title(f'ROC Classe {cls}')
        plt.legend(loc="lower right")

        # Ajustar o layout para evitar sobreposição
    plt.tight_layout()
    plt.savefig(file_name)
    plt.show()

# Gerar o gráfico para as 11 classes
plot_roc_for_eleven_classes(range(11),'ROC.png')

print('Gráficos ROC e AUC feitos.')