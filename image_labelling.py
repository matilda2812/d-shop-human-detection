''' Module to alter images with labels identified by ML
    Labels are customizable
    Author: Matilda Stevenson, Presales Development Intern
    Contact: matilda.stevenson@sap.com
    Date: 7/11/18
'''

import numpy as np
from PIL import Image
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont

# Called if no people are detected
def draw_no_one(image):

    # convert np array to image
    image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
    _draw_text_on_image(image=image_pil, human_count=0)

    np.copyto(image, np.array(image_pil))


# Receive image for drawing boxes on and convert to required PIL format
def draw_boxes(image, ymin, xmin, ymax, xmax, color, thickness, display_str_list, human_count=0):

    image_pil = Image.fromarray(np.uint8(image)).convert('RGB')

    _draw_bounding_box_on_image(image_pil, ymin, xmin, ymax, xmax, color,
                               thickness, display_str_list, human_count)

    np.copyto(image, np.array(image_pil))


# Draw box on image using coordinates
def _draw_bounding_box_on_image(image,
                               ymin,
                               xmin,
                               ymax,
                               xmax,
                               color='red',
                               thickness=4,
                               display_str_list=(), human_count=0):

    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size
    (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                    ymin * im_height, ymax * im_height)
    draw.line([(left, top), (left, bottom), (right, bottom),
               (right, top), (left, top)], width=thickness, fill=color)

    _draw_text_on_image(draw=draw, human_count=human_count)


# Draw people-count text on image
# Takes image or draw object depending on whether image has already been drawn
# on
def _draw_text_on_image(image=None, draw=None, human_count=0):

    if human_count == 0:
         draw = ImageDraw.Draw(image)

    if human_count == 1:
        text = " person currently in the d-shop"
    else:
        text = " people currently in the d-shop"

    text = str(human_count) + text

    if draw != None:
        draw.text((0, 0),text,(255,255,255))
