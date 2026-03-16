import os

import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

MODEL_PATH = "CvsD-Classification.h5"

# Load trained model (if available)
model = None
if os.path.exists(MODEL_PATH):
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        print(f"⚠️ Failed to load model '{MODEL_PATH}': {e}")
else:
    print(f"⚠️ Model file not found at '{MODEL_PATH}'. Please add it to run predictions.")

IMG_SIZE = 128


def predict_image(img):

    # Handle empty input
    if img is None:
        return {"Error": 1.0}

    # Convert image to RGB
    img = img.convert("RGB")

    # Resize image
    img = img.resize((IMG_SIZE, IMG_SIZE))

    # Convert to array
    img_array = np.array(img)

    # Normalize
    img_array = img_array / 255.0

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    if model is None:
        return {"Error": "Model not loaded"}

    try:
        prediction = model.predict(img_array)[0][0]
    except Exception as e:
        return {"Error": str(e)}

    dog_prob = float(prediction)
    cat_prob = float(1 - prediction)

    return {
        "Dog 🐶": dog_prob,
        "Cat 🐱": cat_prob
    }


with gr.Blocks() as demo:

    gr.Markdown(
        """
        # 🐱🐶 Cat vs Dog Image Classifier
        Upload an image and the CNN model will predict whether it is a **Cat** or **Dog**.
        """
    )

    with gr.Row():

        with gr.Column():

            image_input = gr.Image(
                type="pil",
                label="Upload Image"
            )

            predict_btn = gr.Button("Predict")

        with gr.Column():

            output = gr.Label(label="Prediction")

    predict_btn.click(
        fn=predict_image,
        inputs=image_input,
        outputs=output
    )

    gr.Markdown(
        """
        ---
        **Model:** Convolutional Neural Network (CNN)  
        **Dataset:** Kaggle Cats vs Dogs Dataset  
        **Framework:** TensorFlow / Keras   
        **Done By:** Mithun 
        """
    )


demo.launch()