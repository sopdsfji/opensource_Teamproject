<h1>Lane Detector</h1>


<h2>1. 프로젝트 개요</h2>

+ 지난 13일 발표된 도로교통안전 추진 전략에 따르면, 완전 자율주행차의 상용화에 대비해 내년부터 신규 운전면허 취득자를 대상으로 자율주행차 교통안전교육이 시행될 예정이다. 2025년까지 자율주행시스템의 교통법규 위반에 따른 벌점·과태료 등 제재가 정비되고, 특정 수준 이상의 자율주행시스템이 적용된 차종만을 운전할 수 있는 '간소 운전면허'가 2028년 도입된다고 한다. 이러한 변화에 주목하여 우리 조는 자율주행차에 흥미를 느꼈고 간단한 자율주행차의 차선 인식 프로그램을 구현해보았다. 

+ 차선 인식 프로그램은 자율주행차의 안전성을 높이는데 기여하며 차량이 도로의 차선을 정확하게 인식하고 이를 따라갈 수 있도록 돕는다. 또한 차량이 차선을 벗어나지 않도록 하여 교통사고를 예방하고 운전자가 안전하게 운전할 수 있게 한다. 
___
<h2>2. 데모 </h2>

___
<h2>3. 사용한 패키지와 version</h2>

- OpenCV version: 4.8.1
- numpy 
- matlplotlib version : 1.24.4
- H264 코덱

___
<h2>4. 실행 방법</h2>

1. 영상파일 저장후 경로 호출
H264사용.
2. 색상 필터링 및 ROI 설정:
color_filter: 노란색 및 흰색 영역을 필터링하여 강조.
roi: 관심 영역 설정.
3. 평면 왜곡 (Perspective Transformation):
wrapping: 차량의 시야를 편하게 바꾸기 위해 원근 변환 적용.
4. 라인 탐지 및 윈도우 검색:
plothistogram: 히스토그램을 통해 초기 라인 위치 추정.
slide_window_search: 슬라이딩 윈도우 기반으로 라인 탐지 및 추정.
5. 라인 그리기 및 결과 표시:
draw_lane_lines: 탐지된 라인 주위에 영역을 그려 원본 이미지에 표시.
out1.write(result): 동영상 녹화.
6. 무한 루프 및 종료:
동영상 프레임을 읽어와서 각 단계를 수행하고, ESC 키를 누를 때까지 반복.

___
<h2>5. 참고 자료</h2>
*누르면 이동합니다*

https://moon-coco.tistory.com/entry/OpenCV%EC%B0%A8%EC%84%A0-%EC%9D%B8%EC%8B%9D  <h3>:참고 티스토리 링크 </h3> </a>

