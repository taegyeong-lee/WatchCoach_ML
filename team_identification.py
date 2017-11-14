import numpy as np
import cv2
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances


# @brief : 가장 많이 사용된 색을 찾는 함수
# @param : 이미지 또는 이미지주소와 많이 사용된 색 종류 K
# @return : 가장 많이 사용된 색들

def image_color_cluster(image_path, k):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.reshape((image.shape[0] * image.shape[1], 3))
    clt = KMeans(n_clusters=k)
    clt.fit(image)

    return np.int_(clt.cluster_centers_)


# @brief : 팀을 코드를 반환 해주는 함수
# @param : RGB 색, k(색 종류)
# @return : 1 (아군) ,-1 (적군), 0 (기타)

def team_code(rgb, k, standard):
    stadium_rgb = [1, 1, 1]
    our_team_rgb = [1, 1, 1]
    enemy_team_rgb = [1, 1, 1]

    for i in range(0, k):
        if(euclidean_distances([rgb[i]], [stadium_rgb]) > standard):
            return 0;
        if(euclidean_distances([rgb[i]], [our_team_rgb]) > standard):
            return 1;
        if(euclidean_distances([rgb[i]], [enemy_team_rgb]) > standard):
            return -1;


# @brief : 팀을 구별해주는 함수 (메인)
# @param : RGB 색, k(색 종류)
# @return : 1 (아군) ,-1 (적군), 0 (기타)

def team_division(image_path, k, standard):
    best_rgb = image_color_cluster(image_path, k)
    code = team_code(best_rgb, k, standard)
    return code


image_path = "/Users/itaegyeong/Desktop/test2.png"
print(team_division(image_path, 3, 500))








