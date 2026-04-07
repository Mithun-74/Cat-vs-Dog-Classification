import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model("CvsD-Classification.h5")

IMG_SIZE = 128


def predict_image(img):

    if img is None:
        return "Please upload an image", None

    img = img.convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))

    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]

    dog_prob = float(prediction)
    cat_prob = float(1 - prediction)

    label = "Dog 🐶" if dog_prob > cat_prob else "Cat 🐱"

    return label, {"Dog 🐶": dog_prob, "Cat 🐱": cat_prob}


with gr.Blocks() as demo:

    gr.Markdown(
        """
        # 🐱🐶 Cat vs Dog Image Classifier  
        ### Deep Learning Image Classification using CNN

        Upload an image of a **cat or dog**, and the model will predict the class.
        """
    )

    with gr.Row():

        with gr.Column(scale=1):

            image_input = gr.Image(
                type="pil",
                label="Upload Image",
                height=300
            )

            predict_btn = gr.Button("Predict", variant="primary")

        with gr.Column(scale=1):

            prediction_text = gr.Textbox(
                label="Prediction",
                interactive=False
            )

            confidence_chart = gr.Label(
                label="Prediction Confidence"
            )

    predict_btn.click(
        fn=predict_image,
        inputs=image_input,
        outputs=[prediction_text, confidence_chart]
    )

    gr.Markdown(
        """
        ---
        ### 📊 Model Details
        - Model: Convolutional Neural Network (CNN)
        - Dataset: Kaggle Cats vs Dogs
        - Input Size: 128 × 128 RGB images
        - Framework: TensorFlow / Keras

        ### ⚙️ How it Works
        1️⃣ Upload an image  
        2️⃣ Image is resized and normalized  
        3️⃣ CNN predicts probability  
        4️⃣ Result displayed with confidence
        """
    )


demo.launch(theme=gr.themes.Soft())