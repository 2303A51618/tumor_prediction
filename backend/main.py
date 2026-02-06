from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image
import tensorflow as tf
import io
import os

# Initialize FastAPI app
app = FastAPI(title="Neuro Assistant - Brain Tumor Detection")

# Allow frontend access later
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model (deployment-safe Keras .h5 only)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = "brain_tumor_model.h5"
MODEL_FILE = os.path.join(BASE_DIR, MODEL_PATH)

if not os.path.exists(MODEL_FILE):
    raise RuntimeError(
        "Model file not found: 'brain_tumor_model.h5' in backend folder. "
        "Run the notebook cell that saves model.save(\"brain_tumor_model.h5\")."
    )

model = tf.keras.models.load_model(MODEL_FILE)

# Class labels (MUST match training order)
CLASS_LABELS = [
    "Glioma Tumor",
    "Meningioma Tumor",
    "No Tumor",
    "Pituitary Tumor"
]

IMAGE_SIZE = 150


@app.get("/")
def root():
    return {"status": "Neuro Assistant API is running"}


def preprocess_image(image: Image.Image):
    image = image.convert("RGB")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read image file
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    # Preprocess image
    input_tensor = preprocess_image(image)

    # Model prediction
    predictions = model.predict(input_tensor)[0]

    # Get predicted class
    predicted_index = int(np.argmax(predictions))
    predicted_class = CLASS_LABELS[predicted_index]
    confidence = float(predictions[predicted_index])

    # All class probabilities
    all_probs = {
        CLASS_LABELS[i]: float(predictions[i])
        for i in range(len(CLASS_LABELS))
    }

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "all_probabilities": all_probs
    }
