import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from core.functions import *
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

# return interpreter != None if framework == 'tflite', infer != None if framework != 'tflite'
def Initialize(flags_model, flags_weights, flags_framework='tf', flags_tiny=False):
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    print("-> Loading model ", flags_model)

    keeping_alive=[session]
    if flags_framework == 'tflite':
        interpreter = tf.lite.Interpreter(model_path=flags_weights)
        interpreter.allocate_tensors()
        print(interpreter.get_input_details())
        print(interpreter.get_output_details())
    else:
        saved_model_loaded = tf.saved_model.load(flags_weights, tags=[tag_constants.SERVING])
        interpreter = saved_model_loaded.signatures['serving_default']
        keeping_alive.append(saved_model_loaded)

    class_names = read_class_names(cfg.YOLO.CLASSES)
    return interpreter, keeping_alive, class_names

# return boxes, scores, classes, num_objects
def ExtractWithInfer(frame, infer, flag_framework, resize_img_to, flag_iou=0.45, threshold=0.50):
    frame_size = frame.shape[:2]
    image_data = cv2.resize(frame, (resize_img_to, resize_img_to))
    image_data = image_data / 255.
    image_data = image_data[np.newaxis, ...].astype(np.float32)
    start_time = time.time()

    for key, value in (infer(tf.constant(image_data))).items():
        boxes = value[:, :, 0:4]
        pred_conf = value[:, :, 4:]

    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
        scores=tf.reshape(
            pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
        max_output_size_per_class=50,
        max_total_size=50,
        iou_threshold=flag_iou,
        score_threshold=threshold
    )
    
    original_h, original_w, _ = frame.shape
    bboxes = utils.format_boxes(boxes.numpy()[0], original_h, original_w)
    
    data = [bboxes, scores.numpy()[0], classes.numpy()[0], valid_detections.numpy()[0]]
    return data