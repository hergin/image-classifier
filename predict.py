import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Set paths
data_dir = "data"
img_size = (128, 128)
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

# Load model
model = load_model("model.h5")

# Predict
def predict(image_path):
    img = load_img(image_path, target_size=img_size)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions)
    class_labels = list(train_data.class_indices.keys())
    return class_labels[class_idx], predictions[0][class_idx]

# Example usage
label, confidence = predict("data-cropped/dark-with-train/3341875975_20241105005934_IMAG0945-100-945.JPG")
print(f"Predicted: {label} with confidence {confidence}")