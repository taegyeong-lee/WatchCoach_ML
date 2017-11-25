## WatchCoach_ML 

- 소프트웨어 마에스트로 8기 (영상처리, 머신러닝) 부분
- server에 팀코드와 좌표값을 전송



## Feature

- **main.py :** mp4 동영상으로부터 사람인식(object detection, 텐서플로우 기본모델 사용)

- **team_identification.py :** 인식한 사람의 아군(1)과 적군(-1) 그리고 기타(0)로 구별

- **transfer_view.py :** 원래 영상의 카메라에서의 사람의 좌표를 Bird'eyes View(Top View)로 좌표 변환

- **video :** test video

- test/improving_dlib_object_tracking.py : object tracking with dlib

- test/improving_opencv_object_detection.py : object detection with opencv

- test/improving_tensor_object_detection.py : object detection with tensorflow

  ​


```

붉은색 팀을 구별하기 위해서 사용하는 방법

A. 텐서플로우를 사용한 방법 (현재 방법)
사람 인식(텐서플로우) -> 예외처리(비정상적인 크기) -> 색상으로 팀 구별 -> 좌표생성

1. 문제점
- 매 이미지마다 인식을 해 속도느림
- 인식을 못하는 사람 발생
- 겹쳤을때, 갑자기 사라질때 탐지 불가능
- 오탐 발생

2. 해결방안
- 방법 B와 연동, 속도는 더 줄어듦, 정확도 향샹 보장 X

--------------------------------------------

B. opencv 를 사용한 방법 (현재 진행중)
붉은색 검출 -> 잡음제거 -> 붉은색 강조 -> 붉은색 검출 -> 좌표생성

1. 문제점
- 색이 정확하지 않을 경우 탐지 불가
- 색이 겹치거나 할 경우 오탐 발생

2. 해결방안
- 배경제거 -> 움직임 감지 및 테두리 -> 유니폼색 구별 -> 좌표반환
- 겹쳤을때? 갑자기 사라졌을때? 인식 자체가 안될때 ?

--------------------------------------------

축구 동영상으로 진행하였을시 인식 잘됨
농구 동영상, 겹치는 부분이 많으며 유니폼 인식이 잘 안될 경우 영상인식 자체가 거의 불가능

```

## Lib

tensorflow 1.2.1

### 