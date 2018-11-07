''' Detect people using a webcam and SAP Human Detection API
    Author: Matilda Stevenson, Presales Development Intern
    Contact: matilda.stevenson@sap.com
    Date: 7/11/18
'''

import cv2
import os
from api_requests import send_image_from_video
from image_labelling import draw_boxes, draw_no_one
import numpy as np
from PIL import Image

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Initialize webcam feed
video = cv2.VideoCapture(0)
ret = video.set(3,1280)
ret = video.set(4,720)


while(True):
    print("true")
    ret, frame = video.read()
    frame_expanded = np.expand_dims(frame, axis=0)

    img_path = CWD_PATH + '/frame.jpg'
    cv2.imwrite(img_path, frame)
    # get human boxes from Human Detection API
    boxes = send_image_from_video(img_path,"frame.jpg", False)

    # draw boxes and text on image
    if len(boxes) == 0:
        draw_no_one(frame)

    else:
        for box in boxes:
            ymin, xmin, ymax, xmax = box
            draw_boxes(frame, ymin, xmin, ymax, xmax, color='red',
            thickness=8,display_str_list=[], human_count=len(boxes))

    # show video with detections
    cv2.imshow('Human detector', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Clean up
video.release()
cv2.destroyAllWindows()
