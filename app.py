import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import joblib

# Load the pre-trained model
model = joblib.load('model1.sav')

# Load the background image
background_image = Image.open('water.jpg')

# Streamlit UI
st.image(background_image)
st.title("Water Potability Prediction")

st.sidebar.header("Input Parameters")

# Add slider bars for each parameter
ph = st.slider("pH", 0.0, 14.0, 7.0, 0.1)
hardness = st.slider("Hardness (mg/L)", 0.0, 500.0, 100.0, 1.0)
solids = st.slider("Solids (mg/L)", 0.0, 5000.0, 1000.0, 10.0)
chloramines = st.slider("Chloramines (mg/L)", 0.0, 10.0, 4.0, 0.1)
sulfate = st.slider("Sulfate (mg/L)", 0.0, 500.0, 250.0, 1.0)
conductivity = st.slider("Conductivity (uS/cm)", 100.0, 5000.0, 1000.0, 10.0)
organic_carbon = st.slider("Organic Carbon (mg/L)", 0.0, 50.0, 10.0, 1.0)
trihalomethanes = st.slider("Trihalomethanes (ug/L)", 0.0, 200.0, 50.0, 1.0)
turbidity = st.slider("Turbidity (NTU)", 0.0, 10.0, 5.0, 0.1)

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
    prediction = model.predict(input_df)

    st.subheader("Prediction Result:")
    if prediction[0] == 1:
        st.write("The water is potable.")
    else:
        st.write("The water is not potable.")
