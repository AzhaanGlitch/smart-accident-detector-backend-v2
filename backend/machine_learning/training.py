import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

IMG_SIZE = (224, 224, 3)
img_size = (224, 224)
batch_size = 32
DATASET_DIR = r'D:\Smart Accident Detector\ML\Dataset_02' 

train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    DATASET_DIR,
    labels='inferred',
    label_mode='binary',
    class_names=['accident_folder', 'non_accident_folder'], 
    image_size=img_size,
    batch_size=batch_size,
    validation_split=0.2,
    subset='training',
    seed=43
)

validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    DATASET_DIR,
    labels='inferred',
    label_mode='binary',
    class_names=['accident_folder', 'non_accident_folder'],
    image_size=img_size,
    batch_size=batch_size,
    validation_split=0.2,
    subset='validation',
    seed=43
)

# Split validation into validation and test sets
val_batches = tf.data.experimental.cardinality(validation_dataset)
test_dataset = validation_dataset.take(val_batches // 2)
validation_dataset = validation_dataset.skip(val_batches // 2)

train_size = tf.data.experimental.cardinality(train_dataset).numpy() * batch_size
validation_size = tf.data.experimental.cardinality(validation_dataset).numpy() * batch_size
test_size = tf.data.experimental.cardinality(test_dataset).numpy() * batch_size

print(f"Training samples: {train_size}")
print(f"Validation samples: {validation_size}")
print(f"Test samples: {test_size}")

AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

base_model = MobileNetV2(input_shape=IMG_SIZE, include_top=False, weights='imagenet')
base_model.trainable = False  # Freeze base model

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint = tf.keras.callbacks.ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=20,
    callbacks=[early_stopping, model_checkpoint]
)

test_loss, test_acc = model.evaluate(test_dataset)
print(f"Test Accuracy: {test_acc:.2f}")

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.savefig('training_graph.png') 
plt.show()

model.save('accident_detection_model.h5')
print("Model saved as 'accident_detection_model.h5'")