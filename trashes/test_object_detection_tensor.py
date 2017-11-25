import cv2
import os
import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
import time

# local modules
import team_identification as ti
import canvas_show as cs

# 모델 설정
CWD_PATH = os.getcwd()
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
PATH_TO_CKPT = os.path.join(CWD_PATH, 'object_detection', MODEL_NAME, 'frozen_inference_graph.pb')
PATH_TO_LABELS = os.path.join(CWD_PATH, 'object_detection', 'data', 'mscoco_label_map.pbtxt')
NUM_CLASSES = 3

# 라벨 로딩
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# @brief : 물체인식 + 팀구별 후 각 선수들의 위치를 반환하는 함수
# @param : 이미지, 이미지 가로, 이미지 세로, 텐서세션, 텐서그래프
# @return : 물체인식된 이미지, (원래이미지의) 아군선수 좌표, 적군선수 좌표, 기타물체 좌표
def detect_objects(image_np, w, h, sess, detection_graph):
    start = time.time()

    image_np_expanded = np.expand_dims(image_np, axis=0)
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})


    ourTeamPoint = []
    enemyTeamPoint = []
    otherPoint = []


    ourTeamTrackingPoint = []
    enemyTeamTrackingPoint = []
    otherTrackingPoint = []


    for detectionObject, detectionScore, detectionBox in zip(classes, scores, boxes):
        for finalObject, finalScore, finalBoxPoint in zip(detectionObject, detectionScore, detectionBox):
            if finalObject == 1 and finalScore > 0.2:

                # top left, bottom right
                br_point = [int(finalBoxPoint[1] * w), int(finalBoxPoint[0] * h)]
                tl_point = [int(finalBoxPoint[3] * w), int(finalBoxPoint[2] * h)]

                # 비정상적인 크기 예외처리
                if tl_point[0] - br_point[0] > 300 or \
                                        tl_point[0] - br_point[0] < 10 or \
                                        tl_point[1] - br_point[1] > 80:
                    continue

                # 사람 위치
                point = (tl_point[0] + (br_point[0] - tl_point[0]) / 2, tl_point[1])

                # 이미지 몸통만 분리
                cut_point_y1 = int(br_point[1] + (tl_point[1] - br_point[1]) / 4.5)  # 머리부터 목까지
                cut_point_y2 = int(tl_point[1] + (br_point[1] - tl_point[1]) / 2)  # 다리부터 몸통까지
                cut_image = image_np[cut_point_y1:cut_point_y2, br_point[0]:tl_point[0]]


                team_code = ti.team_division(cut_image)

                # 1 은 아군
                if team_code == 1:
                    ourTeamPoint.append(point)
                    ourTeamTrackingPoint.append([br_point[1], tl_point[1], br_point[0], tl_point[0]])

                # -1 은 적군
                elif team_code == -1:
                    enemyTeamPoint.append(point)
                    enemyTeamTrackingPoint.append([cut_point_y1,cut_point_y2, br_point[0], tl_point[0]])

                # 0 은 심판
                elif team_code == 0:
                    otherPoint.append(point)

    # 물체인식 완료된 이미지와 아군/적군 위치 반환
    return image_np, ourTeamTrackingPoint, enemyTeamTrackingPoint, otherPoint


def main_processing(frame, w, h):
    # 그래프 설정하기
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.Session(graph=detection_graph)

    image, our_team_tracking_point, enemy_team_tracking_point, other_point = detect_objects(frame, w, h, sess, detection_graph)

    return our_team_tracking_point,enemy_team_tracking_point,our_team_tracking_point







