import streamlit as st
import numpy as np
from keras.models import load_model
from PIL import Image
from streamlit_drawable_canvas import st_canvas

def main():
    st.title("MNIST Classification App")
    model_path = "model/MNIST_keras_CNN.h5"
    # Load the pre-trained model
    try:
        model = load_model(model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return
    
    tab_file, tab_draw = st.tabs(["File Upload", "Draw Digit"])
    with tab_file:
        st.subheader("Upload an Image of a Digit")
        image = st.file_uploader("Upload an image of a digit (0-9)", type=["png", "jpg", "jpeg"])
        if image is not None:
            # Preprocess the image and make predictions
            img = Image.open(image)
            img = img.convert("L")  # Convert to grayscale
            img = img.resize((28, 28))
            img = np.array(img)
            img = img.reshape(1, 28, 28, 1)
            img = img.astype("float32") / 255

            # Make predictions
            predictions = model.predict(img)
            predicted_class = np.argmax(predictions)
            predicted_confidence = np.max(predictions)

            st.subheader(f"Predicted Class: {predicted_class}")
            st.subheader(f"Prediction Confidence: {predicted_confidence:.2f}")
            st.image(image, caption="Uploaded Image", use_container_width=True)
    with tab_draw:
        st.subheader("Draw a Digit")
        
        # Create a canvas for drawing
        canvas = st_canvas(
            fill_color="white",
            background_color="black",
            stroke_color="white",
            stroke_width=20,
            width=280,
            height=280,
            key="canvas"
        )
        
        if canvas.image_data is not None:
            # Convert the drawn image to a format suitable for prediction
            img = Image.fromarray(canvas.image_data.astype(np.uint8))
            img = img.convert("L")
            img = img.resize((28, 28))
            img = np.array(img)
            img = img.reshape(1, 28, 28, 1)
            img = img.astype("float32") / 255

            # Make predictions
            predictions = model.predict(img)
            predicted_class = np.argmax(predictions)
            predicted_confidence = np.max(predictions)

            st.subheader(f"Predicted Class: {predicted_class}")
            st.subheader(f"Prediction Confidence: {predicted_confidence:.2f}")

if __name__ == "__main__":
    main()
