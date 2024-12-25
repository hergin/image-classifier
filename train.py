import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
import os
from tensorflow.keras.layers import SeparableConv2D

# Set paths
data_dir = "data-cropped-smaller"
img_size = (300, 300)
batch_size = 32

# Load data
datagen = ImageDataGenerator(validation_split=0.2, rescale=1./255)
train_data = datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)
val_data = datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

model = Sequential([
    Conv2D(16, (3, 3), activation='relu', input_shape=(*img_size, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(train_data.num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
# model.fit(train_data, validation_data=val_data, epochs=10)
# Can be set 5 epoch because it is getting steadier
model.fit(train_data, validation_data=val_data, epochs=10)

# Save the model
model.save("model.keras")
print("Model saved as 'model.keras'")
