import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Set paths
data_dir = "data-cropped-smaller"
img_size = (300, 300)
batch_size = 32
class_labels = ['dark-no-train', 'dark-with-train', 'light-no-train', 'light-with-train']

# Load model
model = load_model("model.keras")

# Predict
def predict(image_path):
    global class_labels
    img = load_img(image_path, target_size=img_size)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions)
    class_labels = class_labels
    return class_labels[class_idx], predictions[0][class_idx]

# Example usage
label, confidence = predict("data-cropped-smaller/dark-with-train/3341875975_20241105005934_IMAG0945-100-945.JPG")
print(f"Predicted: {label} with confidence {confidence}")