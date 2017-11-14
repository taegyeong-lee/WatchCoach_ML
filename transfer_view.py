import cv2
import numpy as np

# @brief : 변환 행렬 구하는 함수
# @param : 원래 이미지의 각 꼭짓점 좌표
# @return : 변환행렬, 변환된 이미지의 가로, 세로
def get_trans_matrix(tl, bl, tr, br):

    # 오리지널 이미지
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


# @brief : 각 선수들의 좌표를 (탑뷰로) 변환 하는 함수
# @param : 원래이미지, 우리팀좌표, 적팀좌표, 기타물체좌표, 변환행렬, 변환된 이미지 가로, 변환된 이미지 세로
# @return : 변환된 이미지, 변환된 우리팀 좌표, 변환된 적팀 좌표, 변환된 기타물체 좌표
def trans_object_point(original_image, our_team_point, enemy_team_point, other_point, trans_matrix, trans_image_weight, trans_image_height):

    trans_our_team_point = None
    trans_enemy_team_point = None
    trans_other_point = None

    dst = cv2.warpPerspective(original_image, trans_matrix, (trans_image_weight, trans_image_height))

    # 좌표 변환하고 테스트를 위해 점 찍기
    if our_team_point != []:
        original_1 = np.array([(our_team_point)], dtype=np.float32)
        trans_our_team_point = cv2.perspectiveTransform(original_1, trans_matrix)
        draw_circle(dst, trans_our_team_point,(255, 0, 0))

    if enemy_team_point != []:
        original_2 = np.array([(enemy_team_point)], dtype=np.float32)
        trans_enemy_team_point = cv2.perspectiveTransform(original_2, trans_matrix)
        draw_circle(dst, trans_enemy_team_point,(255, 0, 255))

    if other_point != []:
        original_3 = np.array([(other_point)], dtype=np.float32)
        trans_other_point = cv2.perspectiveTransform(original_3, trans_matrix)
        draw_circle(dst, trans_other_point,(0, 0, 255))

    # 점 찍힌 이미지와 변환된 사람 좌표를 리턴함
    return dst, trans_our_team_point, trans_enemy_team_point, trans_other_point


# @brief : 테스트(시각화)용 점 찍어주는 함수
# @param : 이미지, 좌표, 색
# @return : 없음
def draw_circle(image, point, rgb):
    for i in point:
        for i2 in i:
            cv2.circle(image, (tuple)(i2), 10, rgb, -1)

