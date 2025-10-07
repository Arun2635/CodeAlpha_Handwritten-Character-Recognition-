import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

# -----------------------------
# 1. Load EMNIST dataset
# -----------------------------
# EMNIST is available via tensorflow_datasets or manually
import tensorflow_datasets as tfds

(ds_train, ds_test), ds_info = tfds.load(
    'emnist/letters',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True
)

# -----------------------------
# 2. Preprocess the data
# -----------------------------
def normalize_img(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label-1   # labels go from 1-26, shift to 0-25

ds_train = ds_train.map(normalize_img).batch(128).prefetch(tf.data.AUTOTUNE)
ds_test = ds_test.map(normalize_img).batch(128).prefetch(tf.data.AUTOTUNE)

# -----------------------------
# 3. Build CNN model
# -----------------------------
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(26, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# -----------------------------
# 4. Train the model
# -----------------------------
history = model.fit(ds_train, epochs=5, validation_data=ds_test)

# -----------------------------
# 5. Evaluate the model
# -----------------------------
test_loss, test_acc = model.evaluate(ds_test)
print(f"✅ Test Accuracy: {test_acc:.4f}")

# -----------------------------
# 6. Predict some samples
# -----------------------------
class_names = [chr(i) for i in range(65,91)]  # A-Z

for images, labels in ds_test.take(1):
    preds = model.predict(images)
    plt.figure(figsize=(10,10))
    for i in range(9):
        plt.subplot(3,3,i+1)
        plt.imshow(images[i].numpy().reshape(28,28), cmap="gray")
        plt.title(f"Pred: {class_names[np.argmax(preds[i])]} | True: {class_names[labels[i].numpy()]}")
        plt.axis("off")
    plt.show()
