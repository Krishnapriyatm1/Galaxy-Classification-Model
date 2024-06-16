import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


# Constants
IMG_SIZE = (224, 224)
CLASSES = ['Barred_Spiral','Cigar_Round_Smooth','Disturbed','Edge_on_with_Bulge','Edge_On_Without_Bulge','In_Bettween_Smooth','Merging','Round_Smooth','Un_Barred_Loose_Spiral','UnBarred_Tight_Spiral' ]

# Load model
st.cache(allow_output_mutation=True)
def load_my_model():
    model = tf.keras.models.load_model('Galaxy_New_model.h5')
    return model

# Function to preprocess image
def preprocess_image(img):
    # Convert the PIL image to RGB format
    img_rgb = img.convert('RGB')

    # Resize the image to match the input size expected by the model
    img_resized = img_rgb.resize(IMG_SIZE)

    # Convert the image to an array and normalize the pixel values
    img_array = np.array(img_resized)
    img_normalized = img_array / 255.0  # Normalize pixel values to the range [0, 1]

    # Expand dimensions to match the expected shape for the model
    img_expanded = np.expand_dims(img_normalized, axis=0)

    return img_expanded

# Define main function
def main():
    st.title("Galaxy Classification Model")


    # Create file uploader widget
    uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])

    # Check if image is uploaded
    if uploaded_file is not None:
        # Display uploaded image
        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

        # Preprocess image
        img = preprocess_image(Image.open(uploaded_file))

        # Load model
        model = load_my_model()

        # Get prediction
        prediction = model.predict(img)

        # Get class name
        class_name = CLASSES[np.argmax(prediction)]

        # Display result
        st.write("Prediction: ", class_name)


# Run the app
if __name__ == "__main__":
    main()
