import streamlit as st
import cv2
import numpy as np
from text_placement import find_faces, text_placement_vertical, place_text

st.title('Text Placement App')
st.write("Welcome to my text placement app! Upload your image and specify your desired caption. "
             "This application is designed to detect faces and strategically position the caption to ensure it doesn't "
             "cover anyone's face, providing a seamless and visually appealing result.")


# initialize image upload widget
uploaded_file = st.file_uploader("1) To begin, upload an image")
if uploaded_file is not None:
    # initialize text input widget
    text_input = st.text_input("2) Enter your caption below")
    # Wait for user to enter caption
    if st.button("Caption Image", type="primary"):
        # open image and begin processing
        file_data = np.frombuffer(uploaded_file.read(), np.uint8)
        img = cv2.imdecode(file_data, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        #find faces
        faces = find_faces(img)

        # uses faces (or lack there of) to determine best vertical location of caption
        if len(faces) > 0:
            best_vert_placement = text_placement_vertical(img, faces, engineering_mode=False)
        else:
            im_height, im_width, channels = img.shape
            best_vert_placement = int(im_height * 0.75)

        # place the caption
        processed_img = place_text(text_input, img, best_vert_placement, engineering_mode=False)

        # display final image
        st.image(processed_img, caption='Processed Image', use_column_width=True)

