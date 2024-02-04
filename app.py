import cv2
import numpy as np
import argparse
import streamlit as st

parser = argparse.ArgumentParser()
parser.add_argument("-f","--fname")
parser.add_argument("-t","--text")
args = parser.parse_args()

def find_faces(img):
    '''
    find faces in the image
    :param img: the input image
    :return: an array with size and locations of the faces in pixel space
    '''
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(img_gray)
    return faces

def text_placement_vertical(img, faces, engineering_mode):
    '''
    Find best vertical location for caption box
    Compare the margin between the top of the frame and highest face
    to the margin between the bottom of the frame and the lowest face
    Optimal location is in the middle of the larger of the two margins
    :param img: The input image.
    :param faces: List of detected faces in the format (x, y, width, height).
    :return: The vertical position for optimal text placement.
    '''
    im_height, im_width, channels = img.shape
    min_face_size = im_height / 25
    v_space = [0, im_height]
    face_bbox_top = []
    face_bbox_bot = []
    for x, y, width, height in faces:
        face_bbox_top.append(y - height)
        face_bbox_bot.append(y + height)
        if engineering_mode:
            cv2.rectangle(img, (x, y), (x + width, y + height), color=(255, 0, 0), thickness=2)
            cv2.line(img, pt1=(0, y + height), pt2=(im_width, y + height), color=(0, 255, 0), thickness=1)
            cv2.line(img, pt1=(0, y), pt2=(im_width, y), color=(0, 255, 0), thickness=1)
    margin_top = min(face_bbox_top)
    margin_bot = im_height - max(face_bbox_bot)
    if margin_top > margin_bot:
        best_vert_placement = int(margin_bot / 2)
    else:
        best_vert_placement = int((im_height + max(face_bbox_bot)) / 2)

    # draw
    if engineering_mode:
        cv2.line(img, pt1=(0, best_vert_placement), pt2=(im_width, best_vert_placement), color=(0, 0, 255), thickness=5)

    # Displaying the image
    cv2.imwrite("output/img_annotated.jpg", img)
    return best_vert_placement


def get_optimal_font_scale(text, width, font):
    '''
    Reduce text size until full message fits
    :param text:
    :param width:
    :return: optimal font size
    '''
    for scale in reversed(range(0, 24, 1)):
        textSize = cv2.getTextSize(text, font, fontScale=scale / 10, thickness=1)
        new_width = textSize[0][0]
        if (new_width <= width):
            print(new_width)
            print(textSize)
            return scale / 10

def place_text(text, img, best_vert_placement, engineering_mode):
    '''
    Place caption on the image
    :param text: the caption
    :param img: the input image
    :param best_vert_placement: best location in vertical space for the caption based on face detectionn
    :param engineering_mode:
    :return:
    '''
    im_height, im_width, channels = img.shape
    im_center = int(im_width * 0.5)

    # set width of caption background box
    rect_width = int(0.75 * im_width)  # 50% the width

    # set height of caption background box
    if im_width > im_height:
        rect_height = int(0.1 * im_height)  # 10% the height of landscape images
    else:
        rect_height = int(0.05 * im_height)  # 5% the height of vertical images

    # draw the caption background box
    start_point = (im_center - int(rect_width * 0.5),
                   best_vert_placement - int(rect_height * 0.5))  # represents the top left corner of rectangle
    end_point = (im_center + int(rect_width * 0.5),
                 best_vert_placement + int(rect_height * 0.5))  # represents the top left corner of rectangle
    color = (0, 0, 0)  # Black color in BGR
    rect_thickness = -1  # Thickness of -1 will fill the entire shape
    img = cv2.rectangle(img, start_point, end_point, color, rect_thickness)

    # save the image with rectangles
    cv2.imwrite("output/img_rectangle.jpg", img)

    # add the text
    text_margin_x = int(rect_width * 0.05)
    text_margin_y = int(rect_height * 0.25)
    font = cv2.FONT_HERSHEY_SIMPLEX  # font
    org = (im_center - int(rect_width * 0.5) + text_margin_x,
           best_vert_placement + int(rect_height * 0.5) - text_margin_y)  # text origin
    color = (255, 255, 255)  # Blue color in BGR
    text_thickness = int(im_width / 300)

    font_size = get_optimal_font_scale(text, int(rect_width * 0.9), font)
    cv2.putText(img, text, org, font, font_size, color, text_thickness, cv2.LINE_AA)
    cv2.imwrite("output/img_text.jpg", img)
    return img

######## STREAMLIT INTEGRATION #######
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



