import numpy as np
import cv2
import math
from sklearn.cluster import KMeans

# @brief : 가장 많이 사용된 색을 찾는 함수
# @param : 이미지 또는 이미지주소와 많이 사용된 색 종류 K
# @return : 가장 많이 사용된 색들
def image_color_cluster(image_src, k):
    image2 = cv2.imread(image_src)
    image = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    image = image.reshape((image.shape[0] * image.shape[1], 3))
    clt = KMeans(n_clusters=k)
    clt.fit(image)

    return np.int_(clt.cluster_centers_)


# @brief : 색을 비교해서 비슷한 계열인지 아닌지 판단하는 함수
# @param : 색(RGB), 비교할 색(RGB)
# @return : True (비슷한 색이면), False (비슷한 색이 아니면)
def distance(rgb, standard_rgb):

    r_dist =  math.sqrt((rgb[0] - standard_rgb[0]) * (rgb[0] - standard_rgb[0]))
    g_dist = math.sqrt((rgb[1] - standard_rgb[1]) * (rgb[1] - standard_rgb[1]))
    b_dist = math.sqrt((rgb[2] - standard_rgb[2]) * (rgb[2] - standard_rgb[2]))

    if (r_dist < 50) and (g_dist < 50) and(b_dist < 50):
        return True

    return False


# @brief : 팀을 코드를 반환 해주는 함수
# @param : RGB 색, k(색 종류)
# @return : 1 (아군) ,-1 (적군), 0 (기타)
def team_code(rgb, k):

    stadium_rgb = [0, 0, 255] # blue
    our_team_rgb = [0, 0, 0] # black
    enemy_team_rgb = [255, 0, 0] # red

    # 가장 많이 사용된 색 k 개를 뽑아서, 하나라도 레드, 블루에 속하면 팀 구별해 리턴
    for i in range(0, k):
        if(distance(rgb[i], stadium_rgb)):
            print(rgb[i], "stadium")


        if(distance(rgb[i], our_team_rgb)):
            print(rgb[i], "our team")
            return 1

        if (distance(rgb[i], enemy_team_rgb)):
            print(rgb[i], "enemy team")
            return -1

    print("None")

    return 0


# @brief : 팀을 구별해주는 함수 (메인)
# @param : RGB 색, k(색 종류)
# @return : 1 (아군) ,-1 (적군), 0 (기타)
def team_division(image, k):
    best_rgb = image_color_cluster(image, k)
    print(best_rgb)
    code = team_code(best_rgb, k)
    # print(code)
    return code


image_path = "/Users/itaegyeong/Desktop/testblue.png"
team_division(image_path,3)

