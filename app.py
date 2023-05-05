import cv2
import streamlit as st
import numpy as np
import dark_channel_prior as dcp
import inference as inf

# Function to remove noise from an image
def remove_noise(image):
    # Replace this with your noise removal code
    processed_image, alpha_map = dcp.haze_removal(image, w_size=15, a_omega=0.95, gf_w_size=200, eps=1e-6)
    return processed_image


# Function to perform object detection on an image
def detect_objects(image):
    # Replace this with your object detection code
    # Make sure the output image has bounding boxes around the detected objects
    output_image, class_names = inf.detect(image)
    return output_image, class_names


# Main function for Streamlit app
def app():
    st.title("Underwater Waste Detection model")
    st.text("Upload an image to detect objects")

    # Allow the user to upload an image or video
    file = st.file_uploader("Choose file", type=["jpg", "jpeg", "png"])
    # Process the input and display the output
    if file is not None:
        st.text("Uploading image...")
        input_image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        input_image = cv2.resize(input_image, (416, 416))
        st.text("Input Image:")
        st.image(input_image)

        # Process the input
        st.text("Removing noise from input...")
        processed_image = remove_noise(input_image)
        st.image(processed_image, clamp=True)


        # Run the model
        st.text("Running the model...")
        output_image, class_names = detect_objects(processed_image)

        # Display the output
        st.text("Output Image:")
        # Display "Output Image"
        st.image(output_image)
        if len(class_names)==0:
            st.success("The water is clear!!!")
        else:
            st.error(f"Waste Detected!!!\nThe image has {class_names}")




