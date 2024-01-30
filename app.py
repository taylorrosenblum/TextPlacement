import cv2
import numpy as np
import argparse
import streamlit as st

parser = argparse.ArgumentParser()
parser.add_argument("-f","--fname")
parser.add_argument("-t","--text")
args = parser.parse_args()

def find_faces(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(img_gray)
    # print the number of faces detected
    print(faces)
    print(f"{len(faces)} faces detected in the image.")

    return faces

def text_placement_vertical(img, faces):
    #parameters
    min_face_size = im_height / 25
    v_space = [0, im_height]

    face_bbox_top = []
    face_bbox_bot = []

    # for every face, draw a blue rectangle
    for x, y, width, height in faces:
        if len(faces) >= 10:
            if height > min_face_size:
                cv2.rectangle(img, (x, y), (x + width, y + height), color=(255, 0, 0), thickness=2)
                cv2.line(img, pt1=(0, y + height), pt2=(im_width, y + height), color=(0, 255, 0), thickness=1)
                cv2.line(img, pt1=(0, y), pt2=(im_width, y), color=(0, 255, 0), thickness=1)
            face_bbox_top.append(y - height)
            face_bbox_bot.append(y + height)
        else:
            cv2.rectangle(img, (x, y), (x + width, y + height), color=(255, 0, 0), thickness=2)
            cv2.line(img, pt1=(0, y + height), pt2=(im_width, y + height), color=(0, 255, 0), thickness=1)
            cv2.line(img, pt1=(0, y), pt2=(im_width, y), color=(0, 255, 0), thickness=1)
            face_bbox_top.append(y - height)
            face_bbox_bot.append(y + height)

    margin_top = min(face_bbox_top)
    margin_bot = im_height - max(face_bbox_bot)

    if margin_top > margin_bot:
        center_max_space = int(margin_bot / 2)
    else:
        center_max_space = int((im_height + max(face_bbox_bot)) / 2)

    # draw
    cv2.line(img, pt1=(0, center_max_space), pt2=(im_width, center_max_space), color=(0, 0, 255), thickness=5)

    # Displaying the image
    cv2.imwrite("output/img_annotated.jpg", img)
    return center_max_space

def place_text(text, img, center_max_space):
    # rectangle params
    im_center = int(im_width * 0.5)
    rect_width = int(0.75 * im_width)  # 50% the width

    if im_width > im_height:
        rect_height = int(0.1 * im_height)  # 10% the height of landscape images
    else:
        rect_height = int(0.05 * im_height)  # 5% the height of vertical images


    start_point = (im_center - int(rect_width * 0.5),
                   center_max_space - int(rect_height * 0.5))  # represents the top left corner of rectangle
    end_point = (im_center + int(rect_width * 0.5),
                 center_max_space + int(rect_height * 0.5))  # represents the top left corner of rectangle
    color = (0, 0, 0)  # Black color in BGR
    rect_thickness = -1  # Thickness of -1 will fill the entire shape

    img = cv2.rectangle(img, start_point, end_point, color, rect_thickness)

    # save the image with rectangles
    cv2.imwrite("output/img_rectangle.jpg", img)

    text_margin_x = int(rect_width * 0.05)
    text_margin_y = int(rect_height * 0.25)
    font = cv2.FONT_HERSHEY_SIMPLEX  # font
    org = (im_center - int(rect_width * 0.5) + text_margin_x,
           center_max_space + int(rect_height * 0.5) - text_margin_y)  # text origin
    color = (255, 255, 255)  # Blue color in BGR
    text_thickness = int(im_width / 300)

    def get_optimal_font_scale(text, width):
        for scale in reversed(range(0, 30, 1)):
            textSize = cv2.getTextSize(text, font, fontScale=scale / 10, thickness=1)
            new_width = textSize[0][0]
            if (new_width <= width):
                print(new_width)
                print(textSize)
                return scale / 10
        return 1
    font_size = get_optimal_font_scale(text, int(rect_width * 0.9))
    cv2.putText(img, text, org, font, font_size, color, text_thickness, cv2.LINE_AA)
    cv2.imwrite("output/img_text.jpg", img)
    return img

######## STREAMLIT INTEGRATION #######
st.title('Text Placement App')
st.write("Welcome to this application for adding Instagram-style captions to your images! Start by uploading your image. "
             "Next, provide your desired caption. The app will then analyze your photo, detect faces, and intelligently "
             "position the caption to avoid covering anyone's face.")


# initialize image upload widget
st.subheader("To begin, upload an image")
uploaded_file = st.file_uploader("")
if uploaded_file is not None:
    upload = st.image(uploaded_file, caption='Original Image', use_column_width=True)

    # initialize text input widget
    text_input = st.text_input("Enter your caption below:")

    if st.button("Caption Image", type="primary"):
        # open image and begin processing
        file_data = np.frombuffer(uploaded_file.read(), np.uint8)
        img = cv2.imdecode(file_data, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # initialize the face recognizer (default face haar cascade)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
        print(face_cascade.empty())
        im_height, im_width, channels = img.shape
        faces = find_faces(img)
        if len(faces)==0:
            center_max_space = int(im_height * 0.75)
        else:
            center_max_space = text_placement_vertical(img,faces)
        processed_img = place_text(text_input, img, center_max_space)

        # display final image
        st.image(processed_img, caption='Processed Image', use_column_width=True)



