# This program uses a our TensorFlow-trained neural network to detect vehicles.
# It outputs the vehicle detection parameters for every input image passed on.

## Some of the code is from Google's example at
## https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

## and some from Evan Juras's example at
## https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10/blob/master/Object_detection_image.py

import os
import cv2
import numpy as np
import tensorflow as tf
import sys

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'inference_graph'

# Grab path to current working directory
CWD_PATH = os.path.dirname(os.path.realpath(__file__))

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')
# print(CWD_PATH)
# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')

# Number of classes the object detector can identify
NUM_CLASSES = 4

# Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `5`, we know that this corresponds to `king`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# civilian cars
number_of_cars = 0

# emergency vehicles such as ambulances, and or any other that needs to be added when the model is retrained to be scalable
    ## Due to a lack of time and a mishappening in labelling training images there were not enough emergency vehicle images 
    ## and the like incorporated in training for the model to recognize it.
    ## This is one of the shortcoming of the system at the moment.
number_of_special_permission_vehicle = 0

# pedestrians
number_of_person = 0

label_dict = {1: 'car', 2: 'truck', 3: 'bus', 4: 'person'}

def load_tf_model(image_path):
    # Load the Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.Session(graph=detection_graph)

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

    # Load image using OpenCV and expand image dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_expanded = np.expand_dims(image_rgb, axis=0)

    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: image_expanded})

    global number_of_cars, number_of_special_permission_vehicle, number_of_person
    number_of_cars = 0
    number_of_special_permission_vehicle = 0
    number_of_person = 0

    for index, i in enumerate(scores[0]):
        score = i * 100
        if score >= 90:
            if label_dict[classes[0][index]] == 'car' or label_dict[classes[0][index]] == 'truck' or label_dict[classes[0][index]] == 'bus':
                number_of_cars += 1
            elif label_dict[classes[0][index]] == 'special_vehicle':
                number_of_special_permission_vehicle += 1
            elif label_dict[classes[0][index]] == 'person':
                number_of_person += 1



def get_number_of_cars():
    return number_of_cars

def get_number_of_special_permission_vehicle():
    return number_of_special_permission_vehicle

def get_number_of_persons():
    return number_of_person

def compile_parameters(image_path):
    # first and only command line argument should contain the path to the desired image for vehicle detection
    # otherwise, the function argument is used
    try:
        image_path_ = sys.argv[1]
    except:
        image_path_ = image_path
    load_tf_model(image_path_)
    parameter_dict = {}
    parameter_dict['cars'] = get_number_of_cars()
    parameter_dict['special_permission_vehicles'] = get_number_of_special_permission_vehicle()
    parameter_dict['pedestrians'] = get_number_of_persons()

    return parameter_dict

# Draw the results of the detection (aka 'visulaize the results')

# vis_util.visualize_boxes_and_labels_on_image_array(
#     image,
#     np.squeeze(boxes),
#     np.squeeze(classes).astype(np.int32),
#     np.squeeze(scores),
#     category_index,
#     use_normalized_coordinates=True,
#     line_thickness=8,
#     min_score_thresh=.5)

# All the results have been drawn on image. Now display the image.
# cv2.imshow('Object detector', image)

# Press any key to close the image
# cv2.waitKey(0)

# Clean up
# cv2.destroyAllWindows()