import cv2
import numpy as np


# 탑뷰 변환 행렬을 구하는 함수
# 변환행렬, 변환된이미지가로, 변환된이미지세로 반환
def get_trans_matrix(tl, bl, tr, br):

    # Original Image
    pts1 = np.float32([tl, bl, tr, br])

    # 변환된 이미지의 가로세로 길이 계산
    width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))

    trans_image_weight = max(int(width_a), int(width_b))
    trans_image_height = max(int(height_a), int(height_b))

    # 변환된 새로운 이미지의 가로세로 행렬 만들기
    pts2 = np.array([
        [0, 0],
        [0, trans_image_height - 1],
        [trans_image_weight - 1, 0],
        [trans_image_weight - 1, trans_image_height - 1]], dtype="float32")

    # 변환 행렬
    trans_matrix = cv2.getPerspectiveTransform(pts1, pts2)

    return trans_matrix, trans_image_weight, trans_image_height


# 원래 좌표를 탑뷰 좌표로 변환해주는 함수
# 변환된 이미지와 변환완료된 각각의 좌표 반환
def trans_object_point(original_image, our_team_point, enemy_team_point, other_point, trans_matrix, trans_image_weight, trans_image_height):

    dst = cv2.warpPerspective(original_image, trans_matrix, (trans_image_weight, trans_image_height))

    trans_our_team_point = []
    trans_enemy_team_point = []
    trans_other_point = []


    if our_team_point != []:
        original_1 = np.array([(our_team_point)], dtype=np.float32)
        trans_our_team_point = cv2.perspectiveTransform(original_1, trans_matrix)

    if enemy_team_point != []:
        original_2 = np.array([(enemy_team_point)], dtype=np.float32)
        trans_enemy_team_point = cv2.perspectiveTransform(original_2, trans_matrix)

    if other_point != []:
        original_3 = np.array([(other_point)], dtype=np.float32)
        trans_other_point = cv2.perspectiveTransform(original_3, trans_matrix)


    return dst, trans_our_team_point, trans_enemy_team_point, trans_other_point
