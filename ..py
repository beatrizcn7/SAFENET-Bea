# ----------------- Bibliotecas ---------------
import tensorflow as tf # Principal biblioteca de Machine Learnig e Deep Learning. Cria, treina e implementa modelos de resdes neurais.
from tensorflow.keras import layers, models # API de alto n�vel que facilita a constru��o e treino de redes neurais dentro do TensorFlow
from tensorflow.keras.applications import Xception # Modelo pr�-treinado utilizado
import pandas as pd # Serva para manipulação e análise de dados
import matplotlib.pyplot as plt # Para criar gráficos de diferentes tipos
import seaborn as sns # Para criar gráficos mais elaborados
from sklearn.metrics import confusion_matrix, roc_curve, auc # Usada para calcular a matriz de confusão, e a curva ROC e AUC
import numpy as np # Útil para operações matemáticas
import time  # Biblioteca para contar o tempo
from sklearn.preprocessing import label_binarize # Transforma um array de classes numa matriz binária

# --------------------- Modelo ------------------
base_model = Xception(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

for layer in base_model.layers:
    layer.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(3, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# model.summary()

def parse_tfrecord(example):
    feature_description = {
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        'image/object/class/label': tf.io.FixedLenFeature([], tf.int64),
    }
    parsed_example = tf.io.parse_single_example(example, feature_description)
    image = tf.image.decode_jpeg(parsed_example['image/encoded'], channels=3)
    image = tf.image.resize(image, [224, 224])
    label = parsed_example['image/object/class/label']
    return image, label

def load_dataset(file_paths):
    raw_dataset = tf.data.TFRecordDataset(file_paths)
    dataset = raw_dataset.map(parse_tfrecord)
    return dataset

train_tfrecords = tf.io.gfile.glob('Pasta Final TFRecord - Material/Treino/*.tfrecord')
val_tfrecords = tf.io.gfile.glob('Pasta Final TFRecord - Material/Valida��o/*.tfrecord')

train_dataset = load_dataset(train_tfrecords).batch(32).prefetch(tf.data.AUTOTUNE)
val_dataset = load_dataset(val_tfrecords).batch(32).prefetch(tf.data.AUTOTUNE)

start_time = time.time()

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10
)

# Guardar o modelo treinado
# model.save('Tentativa 1 - Material.h5')
# print('Guardado o modelo!')

test_tfrecords = tf.io.gfile.glob('Pasta Final TFRecord - Material/Teste/*.tfrecord')
test_dataset = load_dataset(test_tfrecords).batch(32).prefetch(tf.data.AUTOTUNE)

test_loss, test_accuracy = model.evaluate(test_dataset)

end_time = time.time()  # Fim
total_time = (end_time - start_time) / 60  # Converter para minutos
# Imprimir o tempo total
print(f"Tempo total: {total_time:.2f} minutos")

# Imprime os resultados da perda e da exatid�o dos dados teste
print(f'Teste Loss: {test_loss}')
print(f'Teste Accuracy: {test_accuracy}')

# ------------- Representação de dados ----------
history_dict = history.history  # Obtém os dados do histórico de treino

# Verifica a existência de val_loss e val_accuracy
data = {
    'Epoch': list(range(1, len(history_dict['loss']) + 1)),  # Número da época
    'Loss': history_dict['loss'],
    'Accuracy': history_dict['accuracy'],
    'Val_Loss': history_dict.get('val_loss', ['N/A'] * len(history_dict['loss'])),
    'Val_Accuracy': history_dict.get('val_accuracy', ['N/A'] * len(history_dict['loss']))
}

# Cria um DataFrame com os dados
df = pd.DataFrame(data)

# Exporta para um arquivo Excel
df.to_excel('Ao longo das Épocas.xlsx', index=False)

print("Arquivo 'Ao longo das Épocas.xlsx' criado com sucesso!")