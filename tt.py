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



img = cv2.imread('/Users/itaegyeong/Desktop/tt.png')
M, w, h = get_trans_matrix([198,251], [16,378], [397, 246], [586,383])
dst = cv2.warpPerspective(img, M, (w, h))
cv2.imshow('original',img)
cv2.imshow('transfer',dst)
cv2.imwrite('transfer.jpg',dst)
cv2.imwrite('original.jpg',img)
cv2.waitKey(0)