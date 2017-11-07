
# @brief : rgb를 비교하여
# @param : 이미지, 이미지 가로, 이미지 세로
# @return : 팀코드 / -1 (기타), 0 (아군), 1 (적군)
def compareRGB(b, g, r, compareRGB):
    count=0
    if b>compareRGB[0][0] and b<compareRGB[1][0]:
        count=count+1
    elif r>compareRGB[0][2] and r<compareRGB[1][2]:
        count = count + 1
    elif g>compareRGB[0][1] and g<compareRGB[1][1]:
        count = count + 1

    if count == 3:
        return True

    return False


# @brief : 하나의 이미지(사람)에 대해 팀을 구별해주는 함수
# @param : 이미지, 이미지 가로, 이미지 세로
# @return : 팀코드 / -1 (기타), 0 (아군), 1 (적군)
def team_identification(image, maxX, maxY):

    stadiumBGR=[(100,100,100),(150,150,150)]
    ourTeamBGR=[(10,10,10),(50,50,50)]
    enemyTeamBGR=[(80,80,80),(99,99,99)]

    ourCount = 0
    enemyCount = 0
    otherCount = 0

    stadiumCount = 0

    for x in range(0, int(maxX)):
        for y in range(0, int(maxY)):

            b = image.item(y, x, 0)
            g = image.item(y, x, 1)
            r = image.item(y, x, 2)

            if compareRGB(b, g, r, stadiumBGR):
                stadiumCount = stadiumCount+1
            else :
                if compareRGB(b, g, r, ourTeamBGR):
                     ourCount = ourCount+1
                elif compareRGB(b, g, r, enemyTeamBGR):
                    enemyCount = enemyCount+1
                else:
                    otherCount = otherCount+1

    if ourCount >= enemyCount:
        return 0
    elif ourCount < enemyCount:
        return 1
    elif otherCount > ourCount and otherCount > enemyCount:
        return -1

