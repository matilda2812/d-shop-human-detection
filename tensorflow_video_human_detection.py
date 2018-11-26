''' Detect people using a video and Google TensorFlow Models
    Based on Google Example from https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb
    Download model from https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
    Choice of model is a speed-accuracy tradeoff
    If using a different model, change MODEL_NAME as required
    Also required in the working directory:
    - https://github.com/tensorflow/models/tree/master/research/object_detection/data (folder and contents)
    - https://github.com/tensorflow/models/tree/master/research/object_detection/utils (folder and contents)
    Author: Matilda Stevenson, Presales Development Intern
    Contact: matilda.stevenson@sap.com
    Date: 26/11/18
'''

# Import packages
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
from image_labelling import draw_on_tf_analysis

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'inference_graph'
VIDEO_NAME = 'person_walking.mp4'

# Grab path to current working directory
CWD_PATH = os.getcwd()
print(CWD_PATH)

# MODEL_NAME = 'faster_rcnn_inception_v2_coco_2018_01_28'
MODEL_NAME = 'ssd_mobilenet_v2_coco_2018_03_29'

# determines how confident prediction must be before label can be applied
SCORE_THRESHOLD = 0.6
# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map
PATH_TO_LABELS = os.path.join(CWD_PATH,'data','mscoco_label_map.pbtxt')

# Path to video
PATH_TO_VIDEO = os.path.join(CWD_PATH,VIDEO_NAME)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

# load label map
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS)

# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Open video file
video = cv2.VideoCapture(PATH_TO_VIDEO)


while(video.isOpened()):

    # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    ret, frame = video.read()
    frame_expanded = np.expand_dims(frame, axis=0)

    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: frame_expanded})

    # print("boxes: ",boxes)
    # print("squeezed boxes: ",np.squeeze(boxes))
    # Draw the results of the detection (aka 'visulaize the results')
    vis_util.visualize_boxes_and_labels_on_image_array(
        frame,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8,
        min_score_thresh=SCORE_THRESHOLD)
    print(np.squeeze(scores))

    human_count = 0
    for score in np.squeeze(scores):
        if score > SCORE_THRESHOLD:
            human_count += 1
    print(human_count)
    draw_on_tf_analysis(frame, human_count)



    # All the results have been drawn on the frame, so it's time to display it.
    cv2.imshow('Object detector', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Clean up
video.release()
cv2.destroyAllWindows()
