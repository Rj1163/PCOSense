# home.py

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image
import tensorflow as tf
import os
from PIL import Image

# Load the pre-trained model
model = tf.keras.models.load_model('bestmodel.keras')


# Define the predict image function
def predictimage(path):
    img = tf.keras.utils.load_img(path, target_size=(224, 224))
    i = tf.keras.utils.img_to_array(img) / 255
    input_arr = np.array([i])
    pred = model.predict(input_arr)
    if pred == 1:
        return "Not Affected"
    else:
        return "Affected"


def app():
    # Set the page layout to "wide" and remove the sidebar

    # Create a container that spans the full width of the page
    container = st.container()

    # Add the "About" section
    container.markdown("""
    <style>
    .about-container {
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        height: 100px;
    }
    .about-text {
        text-align: justify;
    }
    </style>
    <div class="about-container">
        <p class="about-text">
        This app is designed to detect Polycystic Ovary Syndrome (PCOS) using ultrasound images. PCOS is a common hormonal disorder that affects women of reproductive age. It is characterized by irregular menstrual periods, high levels of androgens (male hormones), and polycystic ovaries. This app uses a deep learning model to predict the presence or absence of PCOS in an ultrasound image.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Add a file picker widget
    uploaded_file = container.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    # Display the uploaded image
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        container.image(img, caption="Uploaded image", use_column_width=True)

        # Predict the image
        prediction = predictimage(uploaded_file)
        container.write(f"Prediction: {prediction}")