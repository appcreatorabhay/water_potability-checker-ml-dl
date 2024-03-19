import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import cv2
import numpy as np
import os

# Get the current script's directory and construct the full path to the model file
script_directory = os.path.dirname(__file__)
model_file = os.path.join(script_directory, 'water_cleanliness_model.h5')

# Check if the model file exists
if os.path.exists(model_file):
    # Load the pre-trained model
    model = tf.keras.models.load_model(model_file)
else:
    st.write("Model file not found. Please make sure the model file ('model2.h5') is in the same directory as this script.")

# Define a function to preprocess an image
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(64, 64))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.0
    return img

st.title("Water Cleanliness Detector")

st.write("Upload an image of water to check its cleanliness:")

uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Display the uploaded image
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    # Check if the file has an allowed extension
    if uploaded_image.type == "image/jpeg" or uploaded_image.type == "image/png":
        # Preprocess the image
        image_path = 'uploaded_image.png'
        with open(image_path, "wb") as f:
            f.write(uploaded_image.read())

        img = preprocess_image(image_path)

        # Make a prediction
        prediction = model.predict(img)
        is_clean = prediction[0][0] < 0.5

        # Determine the result
        if is_clean:
            result = "Clean"
        else:
            result = "Dirty"

        st.write(f"Prediction: Water is {result}")
    else:
        st.write("Invalid file format. Please upload a valid image.")
