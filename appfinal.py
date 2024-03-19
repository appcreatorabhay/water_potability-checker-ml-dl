import streamlit as st
import pandas as pd
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import joblib
import numpy as np
from PIL import Image
import cv2

# Define the preprocess_image function to load and preprocess the image
def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((64, 64))  # Resize the image to (64, 64)
    img = np.array(img) / 255.0  # Normalize the image
    img = img.reshape((1, 64, 64, 3))  # Reshape for the model
    return img

# Load the pre-trained water potability model
model1 = joblib.load('model1.sav')

# Get the current script's directory and construct the full path to the image cleanliness model file
script_directory = os.path.dirname(__file__)
model_file = os.path.join(script_directory, 'water_cleanliness_model.h5')

# Check if the image cleanliness model file exists
if os.path.exists(model_file):
    # Load the pre-trained image cleanliness model
    model2 = tf.keras.models.load_model(model_file)
else:
    st.write("Image Cleanliness Model file not found. Please make sure the model file ('water_cleanliness_model.h5') is in the same directory as this script.")

# Streamlit UI
st.title("Water Analysis App")

# Add a radio button for user choice
analysis_option = st.radio("Choose an analysis option:", ("Water Potability", "Water Image Cleanliness"))

if analysis_option == "Water Potability":
    st.title("Water Potability Prediction")

    st.sidebar.header("Input Parameters")

    # Add input fields for each parameter
    ph = st.number_input("pH", 0.0, 14.0, 7.0, 0.1)
    hardness = st.number_input("Hardness (mg/L)", 0.0, 500.0, 100.0, 1.0)
    solids = st.number_input("Solids (mg/L)", 0.0, 5000.0, 1000.0, 10.0)
    chloramines = st.number_input("Chloramines (mg/L)", 0.0, 10.0, 4.0, 0.1)
    sulfate = st.number_input("Sulfate (mg/L)", 0.0, 500.0, 250.0, 1.0)
    conductivity = st.number_input("Conductivity (uS/cm)", 100.0, 5000.0, 1000.0, 10.0)
    organic_carbon = st.number_input("Organic Carbon (mg/L)", 0.0, 50.0, 10.0, 1.0)
    trihalomethanes = st.number_input("Trihalomethanes (ug/L)", 0.0, 200.0, 50.0, 1.0)
    turbidity = st.number_input("Turbidity (NTU)", 0.0, 10.0, 5.0, 0.1)

    # Predict Button
    if st.button("Predict"):
        # Create a dictionary with user input data
        input_data = {
            'ph': ph,
            'Hardness': hardness,
            'Solids': solids,
            'Chloramines': chloramines,
            'Sulfate': sulfate,
            'Conductivity': conductivity,
            'Organic_carbon': organic_carbon,
            'Trihalomethanes': trihalomethanes,
            'Turbidity': turbidity
        }

        # Convert input data to a DataFrame
        input_df = pd.DataFrame([input_data])

        # Make predictions
        prediction = model1.predict(input_df)

        st.subheader("Prediction Result:")
        if prediction[0] == 1:
            st.write("The water is potable.")
        else:
            st.write("The water is not potable.")

elif analysis_option == "Water Image Cleanliness":
    st.title("Water Cleanliness Detector")

    st.write("Choose an image source:")
    image_source = st.radio("Image Source:", ("Upload Image", "Capture Image from Camera"))

    if image_source == "Upload Image":
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
                prediction = model2.predict(img)
                is_clean = prediction[0][0] < 0.5

                # Determine the result
                if is_clean:
                    result = "Clean"
                else:
                    result = "Dirty"

                st.write(f"Prediction: Water is {result}")
            else:
                st.write("Invalid file format. Please upload a valid image.")

    elif image_source == "Capture Image from Camera":
        st.write("Click the 'Capture' button to take a photo from your camera.")
        capture_button = st.button("Capture")

        if capture_button:
            # Initialize the camera (0 is usually the default camera)
            cap = cv2.VideoCapture(0)

            # Check if the camera is opened successfully
            if not cap.isOpened():
                st.write("Error: Could not open the camera.")
            else:
                # Capture a frame from the camera
                ret, frame = cap.read()

                if ret:
                    # Save the captured frame as an image
                    cv2.imwrite("captured_image.jpg", frame)
                    st.write("Image captured successfully.")

                    # Display the captured image
                    st.image("captured_image.jpg", caption="Captured Image", use_column_width=True)

                    # Preprocess the captured image
                    img = preprocess_image("captured_image.jpg")

                    # Make a prediction
                    prediction = model2.predict(img)
                    is_clean = prediction[0][0] < 0.5

                    # Determine the result
                    if is_clean:
                        result = "Clean"
                    else:
                        result = "Dirty"

                    st.write(f"Prediction: Water is {result}")
                else:
                    st.write("Error: Could not capture an image.")

            # Release the camera
            cap.release()
