mono_directory = 'C:/ABSOLUTE_PATH_TO_REPOSITORY/monodepth2'
yolo_directory = 'C:/ABSOLUTE_PATH_TO_REPOSITORY/yolov4'

import cv2
from PIL import Image
import colorsys
import random
import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm
import torch
import math
import bisect
import pyttsx3

import os
import sys
sys.path.append(yolo_directory)
os.chdir(yolo_directory)
import yolov4.implementation as yolo
sys.path.append(mono_directory)
os.chdir(mono_directory)
import monodepth2.implementation as mono

def run_yolov4():
    # YOLO Params
    load_weights = './checkpoints/yolov4-416'
    model = 'yolov4'
    video_path = 0 # string or int
    resize_image_to = 416
    framework = 'tf'
    
    try:
        vid = cv2.VideoCapture(int(video_path))
    except:
        vid = cv2.VideoCapture(video_path)

    infer, keeping_alive, class_names = yolo.Initialize(model, load_weights, framework)
    frame_num = 0
    while True:
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_num += 1
        else:
            print('Video has ended or failed, maybe you can try a different video format!')
            break
        data = yolo.ExtractWithInfer(frame=frame, infer=infer, flag_framework=framework, resize_img_to=resize_image_to)
        boxes, scores, classes, num_objects = data
        
        crop_rate = 20
        if frame_num % crop_rate == 0:
            #create dictionary to hold count of objects for image name
            counts = dict()
            for i in range(num_objects):
                # get count of class for part of image name
                score = scores[i]
                class_index = int(classes[i])
                class_name = class_names[class_index]
                counts[class_name] = counts.get(class_name, 0) + 1
                # get box coords
                xmin, ymin, xmax, ymax = boxes[i]
                print(class_name, str(counts[class_name]), "bounds{", score, "{", xmin, ymin, "} {", xmax, ymax, "}")
        
        cv2.imshow("Camera Input", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cv2.destroyAllWindows()

def run_monodepth2():
    #video_path = 0 # string or int
    video_path = "assets/test_vid4.mp4"
    model = 'mono_1024x320'
    encoder, depth_decoder, device, feed_width, feed_height = mono.Initialize(model)
    
    cap = cv2.VideoCapture(video_path)
    _, frame = cap.read()
    height, width = frame.shape[:2]

    aspect_ratio = float(width)/float(height)
    scaled_feed_height = float(feed_height)*aspect_ratio
    if scaled_feed_height > feed_width :
        feed_aspect = float(feed_width)/float(feed_height)
        crop_height, crop_width = height, int(height*feed_aspect)+1
    else:
        feed_aspect = float(feed_width)/float(feed_height)
        crop_height, crop_width = int(width/feed_aspect), width
    
    pad_height = 100
    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            canvas = frame[:crop_height, :crop_width] #from top corner
            canvas = cv2.resize(canvas, (feed_width, feed_height))
            cv2.imshow(
                'Mirror', 
                mono.GetOutput(canvas, device, encoder, depth_decoder)
            )
            if cv2.waitKey(1) & 0xFF == ord('q'): break
        cv2.destroyAllWindows()

def crop_bounds_for_monodepth(feed_width, feed_height, stock_width, stock_height):
    stock_aspect_ratio = float(stock_width) / float(stock_height)
    scaled_feed_height = float(feed_height)*stock_aspect_ratio
    if scaled_feed_height > feed_width :
        feed_aspect = float(feed_width)/float(feed_height)
        crop_height, crop_width = stock_height, int(stock_height*feed_aspect)+1
    else:
        feed_aspect = float(feed_width)/float(feed_height)
        crop_height, crop_width = int(stock_width/feed_aspect), stock_width
    start_crop_ht = (stock_height-crop_height)/2
    end_crop_ht = start_crop_ht+crop_height
    start_crop_wd = (stock_width-crop_width)/2
    end_crop_wd = start_crop_wd+crop_width
    return int(start_crop_wd), int(start_crop_ht), int(end_crop_wd)+1, int(end_crop_ht)+1

def coords_main_frame_to_mono(x,y, crop_low_x, crop_low_y, crop_to_mono_scale, mono_feed_width, mono_feed_height):
    x-=float(crop_low_x)
    y-=float(crop_low_y)
    x = max(x,0.)
    y = max(y,0.)
    x*=float(crop_to_mono_scale)
    y*=float(crop_to_mono_scale)
    minx = min(x, float(mono_feed_width - 1))
    miny = min(y, float(mono_feed_height- 1))
    return int(minx),int(miny)

def form_sentence_from_data(very_close_objects_on_left,
            very_close_objects_in_front,
            very_close_objects_on_right,
            close_objects_on_left,
            close_objects_in_front,
            close_objects_on_right, flag_info=False):
    #very close ones
    vc_str_init = "there is a "
    vc_str_mid = "very close to you in your "
    c_str_init = "there is a "
    c_str_mid  = "in your "
    formed_sentence_is_empty = True
    formed_sentence = ""
    if len(very_close_objects_in_front) != 0:
        formed_sentence = vc_str_init
        for obj in very_close_objects_in_front:
            formed_sentence += (obj + ", ")
        formed_sentence += (vc_str_mid + "front ")
        formed_sentence_is_empty = False
    
    if len(very_close_objects_on_left) != 0:
        if not formed_sentence_is_empty:
            formed_sentence += "also "
        formed_sentence += vc_str_init
        for obj in very_close_objects_on_left:
            formed_sentence += (obj + ", ")
        formed_sentence += (vc_str_mid + "left ")
        formed_sentence_is_empty = False
    
    if len(very_close_objects_on_right) != 0:
        if not formed_sentence_is_empty:
            formed_sentence += "also "
        formed_sentence += vc_str_init
        for obj in very_close_objects_on_right:
            formed_sentence += (obj + ", ")
        formed_sentence += (vc_str_mid + "right ")
        formed_sentence_is_empty = False
    
    if len(close_objects_in_front) != 0:
        if not formed_sentence_is_empty:
            formed_sentence += "also "
        formed_sentence += c_str_init
        for obj in close_objects_in_front:
            formed_sentence += (obj + ", ")
        formed_sentence += (c_str_mid + "front ")
        formed_sentence_is_empty = False
    
    if len(close_objects_on_right) != 0:
        if not formed_sentence_is_empty:
            formed_sentence += "also "
        formed_sentence += c_str_init
        for obj in close_objects_on_right:
            formed_sentence += (obj + ", ")
        formed_sentence += (c_str_mid + "right ")
        formed_sentence_is_empty = False
    
    if len(close_objects_on_left) != 0:
        if not formed_sentence_is_empty:
            formed_sentence += "also "
        formed_sentence += c_str_init
        for obj in close_objects_on_left:
            formed_sentence += (obj + ", ")
        formed_sentence += (c_str_mid + "left ")
        formed_sentence_is_empty = False

    if not formed_sentence_is_empty:
        if flag_info:
            print(formed_sentence)
    return formed_sentence

def main(debug_output=True):
    # shared
    #video_path = 0 # string or int
    video_path = "C:/ABSOLUTE_PATH_TO_REPOSITORY/assets/test_vid4.mp4"
    video_FOV = 120 # degrees
    try:
        vid = cv2.VideoCapture(int(video_path))
    except:
        vid = cv2.VideoCapture(video_path)
    
    original_height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
    original_width  = vid.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
    print(original_height, original_width, video_path)
    orignal_aspect_ratio = original_width/original_height
    # initialize yolo
    os.chdir(yolo_directory)
    load_weights = './checkpoints/yolov4-416'
    model = 'yolov4'
    yolo_resize_image_to = 416
    yolo_framework = 'tf'
    yolo_infer, yolo_keeping_alives, yolo_class_names = yolo.Initialize(model, load_weights, yolo_framework)
    yolo_num_classes = len(yolo_class_names)

    # initialize mono
    os.chdir(mono_directory)
    model = 'mono_1024x320'
    mono_encoder, mono_depth_decoder, mono_device, mono_feed_width, mono_feed_height = mono.Initialize(model)
    mono_min_dist, mono_max_dist = 3, 50
    
    mono_start_crop_x, mono_start_crop_y, mono_end_crop_x, mono_end_crop_y = crop_bounds_for_monodepth(mono_feed_width, mono_feed_height, original_width, original_height)
    scale_crop_to_mono = float(mono_feed_width)/float(mono_end_crop_x-mono_start_crop_x)

    # initialize pyTTSx3
    tts_engine = pyttsx3.init()
    tts_engine.startLoop(False)

    frame_num = int(0)
    with torch.no_grad():
        hsv_tuples = [(1.0 * x / yolo_num_classes, 1., 1.) for x in range(yolo_num_classes)]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

        random.seed(0)
        random.shuffle(colors)
        random.seed(None)

        previously_formed_sentence = ""
        while True:
            ret, frame = vid.read()
            if not ret:
                print('Video has ended or failed, maybe you can try a different video format!')
                break
            else:
                frame_num += 1
                frame = cv2.flip(frame, 1)
                
            #pre-process
            frame_yolo = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_mono = cv2.resize(
                frame[mono_start_crop_y:mono_end_crop_y, mono_start_crop_x:mono_end_crop_x],
                (mono_feed_width, mono_feed_height)
            )

            # extract yolo output / Object detection
            boxes, scores, classes, num_objects = yolo.ExtractWithInfer(frame=frame_yolo, infer=yolo_infer, flag_framework=yolo_framework, resize_img_to=yolo_resize_image_to)
            
            # extract mono output
            depth_2darr = mono.ExtractDepthTensorFromFrame(frame=frame_mono, device=mono_device, encoder=mono_encoder, depth_decoder=mono_depth_decoder).numpy()
            vmin = np.percentile(depth_2darr, 2)
            vmax = np.percentile(depth_2darr, 98)
            
            # create dictionary to hold count of class_names objects for image name
            counts = dict()
            scene_concerned_close = []
            scene_concerned_very_close = []
            for i in range(num_objects):
                # get count of class for part of image name
                class_index = int(classes[i])
                if debug_output:
                    score = scores[i]
                    class_name = yolo_class_names[class_index]
                    counts[class_name] = counts.get(class_name, 0) + 1
                # get box coords
                xmin, ymin, xmax, ymax = boxes[i]
                
                main_x_l, main_y_l = int((xmin*0.75)+(xmax*0.25)), int((ymin*0.75)+(ymax*0.25))
                main_x_h, main_y_h = int((xmin*0.25)+(xmax*0.75)), int((ymin*0.25)+(ymax*0.75))

                mono_x_l, mono_y_l = coords_main_frame_to_mono(main_x_l, main_y_l, mono_start_crop_x, mono_start_crop_y, scale_crop_to_mono, mono_feed_width, mono_feed_height)
                mono_x_h, mono_y_h = coords_main_frame_to_mono(main_x_h, main_y_h, mono_start_crop_x, mono_start_crop_y, scale_crop_to_mono, mono_feed_width, mono_feed_height)
                
                mid_y, mid_x = int((mono_y_l+mono_y_h)/2.), int((mono_x_l+mono_x_h)/2.)
                depth1 = 0.15*depth_2darr[mono_y_l, mono_x_l]
                depth2 = 0.15*depth_2darr[mono_y_l, mono_x_h]
                depth3 = 0.40*depth_2darr[mid_y, mid_x]
                depth4 = 0.15*depth_2darr[mono_y_h, mono_x_l]
                depth5 = 0.15*depth_2darr[mono_y_h, mono_x_h]
                depth = depth1 + depth2 + depth3 + depth4 + depth5

                limit_ = (mid_x/float(mono_feed_width)) - 0.5
                limit = math.sqrt((0.4 + ((limit_*limit_)/1.5))*0.5)
                
                if depth > limit:
                    scene_concerned_very_close.append([class_index, limit_])
                else:
                    limit = math.sqrt((0.25 + ((limit_*limit_)/1.5))*0.5)
                    if depth > limit:
                        scene_concerned_close.append([class_index, limit_])

                
                if debug_output:
                    cv2.putText(frame, ".", (main_x_l, main_y_l), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 128, 0), 4)
                    cv2.putText(frame, ".", (main_x_h, main_y_l), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 128, 0), 4)
                    cv2.putText(frame, ".", (int((main_x_l+main_x_h)/2.), int((main_y_l+main_y_h)/2.)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 128, 0), 4)
                    cv2.putText(frame, ".", (main_x_l, main_y_h), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 128, 0), 4)
                    cv2.putText(frame, ".", (main_x_h, main_y_h), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 128, 0), 4)

                    cv2.putText(depth_2darr, ".", (mono_x_l, mono_y_l), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 4)
                    cv2.putText(depth_2darr, ".", (mono_x_h, mono_y_l), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 4)
                    cv2.putText(depth_2darr, ".", (int((mono_x_l+mono_x_h)/2.), int((mono_y_l+mono_y_h)/2.)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 4)
                    cv2.putText(depth_2darr, ".", (mono_x_l, mono_y_h), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 4)
                    cv2.putText(depth_2darr, ".", (mono_x_h, mono_y_h), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 4)
                    
                    # Render
                    fontScale = 0.5
                    bbox_color = colors[class_index]
                    bbox_thick = int(0.6 * (original_height + original_width) / 600)
                    cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), bbox_color, bbox_thick)
                    bbox_mess = '%.2f, %.4f' % (limit_, depth)
                    t_size = cv2.getTextSize(bbox_mess, 0, fontScale, thickness=bbox_thick // 2)[0]
                    c3 = (xmin + t_size[0], ymin - t_size[1] - 3)
                    cv2.rectangle(frame, (int(xmin), int(ymin)), (int(c3[0]), int(c3[1])), bbox_color, -1) #filled

                    cv2.putText(frame, bbox_mess, (int(xmin), int(ymin - 2)), cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale, (0, 0, 0), bbox_thick // 2, lineType=cv2.LINE_AA)
            
            very_close_objects_on_left = []
            very_close_objects_in_front = []
            very_close_objects_on_right = []
            close_objects_on_left = []
            close_objects_in_front = []
            close_objects_on_right = []
            for class_index, limit in scene_concerned_close:
                if limit < -0.1:
                    bisect.insort(close_objects_on_left, yolo_class_names[class_index])
                elif limit > 0.1:
                    bisect.insort(close_objects_on_right, yolo_class_names[class_index])
                else:
                    bisect.insort(close_objects_in_front, yolo_class_names[class_index])
            for class_index, limit in scene_concerned_very_close:
                if limit < -0.1:
                    bisect.insort(very_close_objects_on_left, yolo_class_names[class_index])
                elif limit > 0.1:
                    bisect.insort(very_close_objects_on_right, yolo_class_names[class_index])
                else:
                    bisect.insort(very_close_objects_in_front, yolo_class_names[class_index])
            
            sentence = form_sentence_from_data(
                very_close_objects_on_left, very_close_objects_in_front, very_close_objects_on_right, 
                close_objects_on_left, close_objects_in_front, close_objects_on_right, 
                debug_output)
            
            if not (sentence == previously_formed_sentence):
                previously_formed_sentence = sentence
                if len(previously_formed_sentence) != 0:
                    tts_engine.say(previously_formed_sentence)
                    #print("will be said !")

            cv2.imshow("Output1", frame)
            cv2.imshow("Output2", depth_2darr)

            key = cv2.waitKey(1)
            tts_engine.iterate()
            if key & 0xFF == ord('q'): 
                break

        cv2.destroyAllWindows()
        tts_engine.endLoop()



if __name__ == '__main__':
    try:
        #run_yolov4()
        #run_monodepth2()
        main()
    except SystemExit:
        pass