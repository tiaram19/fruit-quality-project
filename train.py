import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Path dataset
dataset_path = "Quality Dataset"
train_path = os.path.join(dataset_path, "train")
valid_path = os.path.join(dataset_path, "valid")
test_path = os.path.join(dataset_path, "test")

# Cek apakah folder ada
for path in [train_path, valid_path, test_path]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"⚠️ Folder {path} tidak ditemukan!")

print("✅ Semua folder ditemukan, lanjut training...")

# Data augmentation & preprocessing
datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, zoom_range=0.2, horizontal_flip=True)
train_generator = datagen.flow_from_directory(train_path, target_size=(150, 150), batch_size=32, class_mode='binary')
valid_generator = datagen.flow_from_directory(valid_path, target_size=(150, 150), batch_size=32, class_mode='binary')

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(test_path, target_size=(150, 150), batch_size=32, class_mode='binary')

# Model CNN
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training
EPOCHS = 10
history = model.fit(train_generator, validation_data=valid_generator, epochs=EPOCHS)

# Simpan model
model.save("fruit_quality_model.h5")
print("✅ Model berhasil disimpan!")
