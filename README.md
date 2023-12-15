# sleepy eyes detector
<프로젝트 개요>
>> 한국도로공사가 밝힌 자료에 따르면 최근 5년 간 12월 사망자 수 연중 가장 높았다. 우리 조는 해당 뉴스를 접하고 졸음 운전 문제에 도움이 될 만한 기술을 구현해보고 싶어 sleepy eyes detector 프로젝트를 진행하게 되었다.
>> 이 프로그램은 실시간 영상과 영상 처리 기술을 기반으로 해당 객체가 졸고 있는 상태인지 깨어있는 상태인지를 파악한다. '눈이 특정 시간 이상 감겨있다-> 조는 상태'
>> dlib를 사용하여 눈으로 판단되는 좌표 값을 얻어내고 (opencv+dlib) 해당 눈의 상태를 파악하는 순서로 구성했다. 눈이 감겼는지 판단하는 classification 부분은 간단한 cnn을 사용했다.

<사용한 패키지, version>
Language: python 3.6.8
CUDA: 10.2

Library
pytorch: 1.5.0
torchvision: 0.6.0
numpy: 1.18.5
opencv-python: 4.2.0.34
dlib: 19.20.0
matplotlib: 3.2.1
imutils: 0.5.3

<실행 방법>
https://github.com/sopdsfji/opensource_Teamproject/blob/main/img.jpg)https://github.com/sopdsfji/opensource_Teamproject/blob/main/img.jpg
](https://github.com/sopdsfji/opensource_Teamproject/blob/main/img.jpg)https://github.com/sopdsfji/opensource_Teamproject/blob/main/img.jpg
