import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Load the pre-trained model
model = load_model("pneumonia_model.h5")

# Streamlit app
st.title("Pneumonia Detection from Chest X-Rays")

uploaded_file = st.file_uploader("Upload a Chest X-Ray Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Save the uploaded file to a temporary directory
    with open("uploaded_image.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Preprocess the image
    img = image.load_img("uploaded_image.jpg", target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the image

    # Make a prediction
    prediction = model.predict(img_array)[0][0]

    # Display the result
    if prediction > 0.5:
        st.error("The model predicts that the patient has **PNEUMONIA**.")
    else:
        st.success("The model predicts that the patient is **NORMAL**.")

