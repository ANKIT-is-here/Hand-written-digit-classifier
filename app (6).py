import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps

# Load model
model = load_model("Ankit_cnn_model.keras")

# App title and description
st.title("✏️ Ankit's Handwritten Digit Recognizer")
st.markdown("Please upload only **28x28 grayscale** digit image and get its prediction.")

# Upload image
uploaded_file = st.file_uploader("Upload a digit image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")  # Convert to grayscale
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # ✅ Replace deprecated ANTIALIAS with Resampling.LANCZOS
    img_resized = ImageOps.fit(image, (28, 28), Image.Resampling.LANCZOS)
    
    img_array = np.array(img_resized)
    img_array = img_array.reshape(1, 28, 28, 1)
    img_array = img_array / 255.0

    # Predict
    prediction = model.predict(img_array)
    predicted_label = np.argmax(prediction)

    st.markdown(f"### 🧠 Predicted Digit: `{predicted_label}`")
    st.markdown(f"**Confidence:** {np.max(prediction)*100:.2f}%")
