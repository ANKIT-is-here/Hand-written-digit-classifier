import streamlit as st
import numpy as np
from tensorflow import keras
from PIL import Image

# Load the pre-trained model
model = keras.models.load_model("Ankit_cnn_model.keras")

# Streamlit app title and instructions
st.title("✏️ Ankit's Handwritten Digit Recognizer")
st.markdown("Upload a **28x28 grayscale** digit image. Model is trained on white digits over black background (like MNIST).")

# Image uploader
uploaded_file = st.file_uploader("Upload a digit image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # Convert image to grayscale and show it
    image = Image.open(uploaded_file).convert("L")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Resize to 28x28 without distortion
    image = image.resize((28, 28))

    # Convert to NumPy array and preprocess
    img_array = np.array(image).astype("float32")
    
    # Invert if background is white (optional, can disable if not needed)
    img_array = 255 - img_array

    # Normalize pixel values and reshape for dense model
    img_array /= 255.0
    img_array = img_array.reshape(1, 784)

    # Predict
    prediction = model.predict(img_array)
    predicted_label = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    # Display result
    st.markdown(f"### 🧠 Predicted Digit: `{predicted_label}`")
    
