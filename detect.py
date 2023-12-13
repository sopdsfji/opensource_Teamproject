#detect에 사용되는 라이브러리 선언
import cv2
import dlib
import numpy as np
from model import Net
import torch
from imutils import face_utils

#얼굴 이미지에서 눈으로 인식하여 크롭한 두 개의 이미지에 대한 사이즈
IMG_SIZE = (34,26)
PATH = './weights/trained.pth'
#dlib 라이브러리에서 얼굴을 인식할 수 있는 변수 선언
detect_face = dlib.get_frontal_face_detector()
predict_face = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
#눈 떴는지 감았는지 판단하는 변수 선언
model = Net()
model.load_state_dict(torch.load(PATH))
model.eval()
#프레임 저장 변수
n_count = 0

#영상이나 사진에서 눈만을 잘라내는 함수
def crop_eye(img, eye_points):
  x1, y1 = np.amin(eye_points, axis=0)
  x2, y2 = np.amax(eye_points, axis=0)
  cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

  w = (x2 - x1) * 1.2
  h = w * IMG_SIZE[1] / IMG_SIZE[0]

  margin_x, margin_y = w / 2, h / 2

  min_x, min_y = int(cx - margin_x), int(cy - margin_y)
  max_x, max_y = int(cx + margin_x), int(cy + margin_y)

  eye_rect = np.rint([min_x, min_y, max_x, max_y]).astype(np.int)

  eye_img = gray[eye_rect[1]:eye_rect[3], eye_rect[0]:eye_rect[2]]

  return eye_img, eye_rect

def predict(pred):
  pred = pred.transpose(1, 3).transpose(2, 3)

  outputs = model(pred)

  pred_tag = torch.round(torch.sigmoid(outputs))

  return pred_tag

while cap.isOpened():
  ret, img_ori = cap.read()

  if not ret:
    break

  img_ori = cv2.resize(img_ori, dsize=(0, 0), fx=0.5, fy=0.5)

  img = img_ori.copy()
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  faces = detector(gray)

  for face in faces:
    shapes = predictor(gray, face)
    shapes = face_utils.shape_to_np(shapes)
    
    eye_img_l, eye_rect_l = crop_eye(gray, eye_points=shapes[36:42])
    eye_img_r, eye_rect_r = crop_eye(gray, eye_points=shapes[42:48])
    eye_img_l = cv2.resize(eye_img_l, dsize=IMG_SIZE)
    eye_img_r = cv2.resize(eye_img_r, dsize=IMG_SIZE)
    eye_img_r = cv2.flip(eye_img_r, flipCode=1)

    eye_input_l = eye_img_l.copy().reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32)
    eye_input_r = eye_img_r.copy().reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32)

    eye_input_l = torch.from_numpy(eye_input_l)
    eye_input_r = torch.from_numpy(eye_input_r)

    pred_l = predict(eye_input_l)
    pred_r = predict(eye_input_r)

    if pred_l.item() == 0.0 and pred_r.item() == 0.0:
      n_count+=1
    else:
      n_count = 0

    if n_count > 100:
      cv2.putText(img,"Sleeping", (120,160), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)
#비디오 소스(cap)로부터 프레임을 캡처하는 루프에 진입
#프레임을 크기 조정하고 처리를 위한 복사본을 만듭니다. 복사본을 흑백으로 변환
#흑백 이미지에서 얼굴을 감지하기 위해 얼굴 검출기를 사용
#각각의 검출된 얼굴에 대해 모양 예측기를 사용하여 얼굴의 랜드마크를 얻음
#왼쪽과 오른쪽 눈 영역을 잘라내고 크기를 조정하며, 오른쪽 눈 이미지를 수평으로 뒤집기
#신경망을 사용하여 각 눈이 열려있는지 아니면 감겨있는지를 예측
#예측에 기반하여 n_count 변수를 업데이트
#눈이 100프레임 이상 연속으로 감겨있을 경우, 프레임에 "Sleeping"을 표시
  cv2.imshow('result', img)
  if cv2.waitKey(1) == ord('x'):
    break
#"X" 누르면 루프에서 빠져나옴
