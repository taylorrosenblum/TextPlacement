import cv2
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-f","--fname", help='path to the photo you want to caption')
parser.add_argument("-t","--text", help='text to put in the caption')
parser.add_argument("-e","--engineering_mode", action='store_true', help='Display alignment features (default: False)')
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
        best_vert_placement = int(margin_top)
    else:
        best_vert_placement = int((im_height + max(face_bbox_bot)) / 2)

    # draw
    if engineering_mode:
        cv2.line(img, pt1=(0, best_vert_placement), pt2=(im_width, best_vert_placement), color=(0, 0, 255), thickness=5)

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

    # set width and height of caption background box
    rect_width = int(0.75 * im_width)  # 50% the width
    rect_height = 60 # not sure why this works

    # draw the caption background box
    start_point = (im_center - int(rect_width * 0.5),
                   best_vert_placement - int(rect_height * 0.5))  # represents the top left corner of rectangle
    end_point = (im_center + int(rect_width * 0.5),
                 best_vert_placement + int(rect_height * 0.5))  # represents the top left corner of rectangle
    color = (0, 0, 0)  # Black color in BGR
    rect_thickness = -1  # Thickness of -1 will fill the entire shape
    img = cv2.rectangle(img, start_point, end_point, color, rect_thickness)

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


# bring in the photo
img = cv2.imread(args.fname)
if img is None:
    print('image did not load')
else:
    # initialize the face recognizer (default face haar cascade)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
    print(face_cascade.empty())
    im_height, im_width, channels = img.shape
    faces = find_faces(img)
    if len(faces)==0:
        center_max_space = int(im_height * 0.75)
    else:
        center_max_space = text_placement_vertical(img,faces,engineering_mode=args.engineering_mode)
    place_text(args.text, img, center_max_space,engineering_mode=args.engineering_mode)

