from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# Load model once
model = load_model('road_sign_model.h5')

def preprocess(image: Image.Image):
    image = image.convert('RGB')
    image = image.resize((30, 30))
    img_array = np.array(image) / 255.0  # Normalize
    return np.expand_dims(img_array, axis=0)

def predict(image: Image.Image):
    img = preprocess(image)
    predictions = model.predict(img)
    return np.argmax(predictions, axis=1)
