import cv2
import multiprocessing
import time
import os
import numpy as np
import tensorflow as tf
from utils.app_utils import FPS
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from moviepy.editor import *
from datetime import datetime


# local modules
import team_identification as ti
import transfer_view as tv


# 모델 설정
MODEL_NAME = 'basic_model'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
CWD_PATH = os.getcwd()
PATH_TO_CKPT = os.path.join(CWD_PATH, 'object_detection', MODEL_NAME, 'frozen_inference_graph.pb')
PATH_TO_LABELS = os.path.join(CWD_PATH, 'object_detection', 'data', 'mscoco_label_map.pbtxt')
NUM_CLASSES = 3


label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# 객체인식해서 사람의 좌표 반환
def detect_objects(image_np, w, h, sess, detection_graph):

    image_np_expanded = np.expand_dims(image_np, axis=0)
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})

    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        min_score_thresh=.4,
        use_normalized_coordinates=True,
        line_thickness=8)

    ourTeamPoint = []
    enemyTeamPoint = []
    otherPoint = []

    for detectionObject, detectionScore, detectionBox in zip(classes, scores, boxes):
        for finalObject, finalScore, finalBoxPoint in zip(detectionObject, detectionScore, detectionBox):
            if finalObject == 1 and finalScore > 0.5:

                personXPoint = int(finalBoxPoint[3] * w)
                personYPoint = int(finalBoxPoint[2] * h)
                point = (personXPoint, personYPoint)

                boxX2Point=int(finalBoxPoint[3] * w)
                boxY2Point=int(finalBoxPoint[2] * h)
                boxX1Point=int(finalBoxPoint[1] * w)
                boxY1Point=int(finalBoxPoint[0] * h)

                # 아군 적군 구별
                cut_image = image_np[boxY1Point:int(boxY2Point*.8)+2, boxX1Point:boxX2Point]
                team_code = ti.teamCutting(cut_image,boxX2Point-boxX1Point,int(boxY2Point*.8)+2-boxY1Point)

                # 0은 아군
                if team_code == 0:
                    ourTeamPoint.append(point)
                # 1은 적군
                elif team_code == 1:
                    enemyTeamPoint.append(point)
                # -1은 심판
                elif team_code == -1:
                    otherPoint.append(point)


    # 물체인식 완료된 이미지와 아군/적군 위치 반환
    return image_np, ourTeamPoint, enemyTeamPoint, otherPoint


def mainProcessing():

    # 그래프 설정하기
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.Session(graph=detection_graph)

    # 버드아이뷰 변환 행렬 구하기
    tl = (480, 494)
    bl = (878, 1036)
    tr = (792, 469)
    br = (1328, 743)
    trans_matrix = tv.get_trans_matrix(tl, bl, tr, br)

    # 비디오 변환
    my_clip = VideoFileClip("/Users/itaegyeong/Desktop/무제 폴더/GOPR0008.MP4")
    w = my_clip.w
    h = my_clip.h
    video = cv2.VideoWriter('/Users/itaegyeong/Desktop/tensorflowvideo3.mp4', -1, 30, (w, h))

    a = 0
    for frame in my_clip.iter_frames():
        a += 1
        if a%10 != 0:
            continue

        image, our_team_point, enemy_team_point, other_point = detect_objects(frame, w, h, sess, detection_graph)
        trans_image, our_trans_team_point, enemy_trans_team_point, trans_other_point = tv.trans_object_point(image, our_team_point, enemy_team_point, other_point, trans_matrix)

        video.write(image)
        image = cv2.resize(image, (480, 270), interpolation=cv2.INTER_CUBIC)
        cv2.imshow('asd', image)
        cv2.imshow('title',trans_image)
        cv2.waitKey(1)

        if a == 300000:
            break

    cv2.destroyAllWindows()
    video.release()






mainProcessing()
