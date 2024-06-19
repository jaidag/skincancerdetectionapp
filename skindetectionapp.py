import streamlit as st
import numpy as np
from PIL import Image
import joblib

# Load the trained KNN model
model_file_path = 'skin_cancer_model.pkl'
model = joblib.load(model_file_path)

# Function to preprocess the image
def preprocess_image(image):
    # Resize the image to a fixed size (e.g., 128x128)
    size = (224, 224)
    image = image.resize(size)
    # Convert the image to a numpy array
    image_array = np.array(image)
    # Flatten the image array
    image_flattened = image_array.flatten()
    return image_flattened


def main():
    st.title("Skin Cancer Detection App")

    st.write("Please note that this app is not 100% accurate. If your result happens to be malignant, please contact a medical professional for further instructions.")

    st.write("**Upload an image to detect if it's benign or malignant:**")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Preprocess the image
        processed_image = preprocess_image(image)

        # Make prediction
        prediction = model.predict([processed_image])[0]
        class_names = {0: 'Benign', 1: 'Malignant'}
        result = class_names[prediction]

        st.write(f"Prediction: {result}")
    st.write("**Please click the button if you have had a large amount of sun exposure:**")
    result = st.button("Yes")
    if result:
        st.write("Please note that high levels of sun exposure can lead to a higher risk of skin cancer.")

if __name__ == '__main__':
    main()