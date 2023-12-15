import cv2
import dlib
import numpy as np
from model import Net
import torch
from imutils import face_utils

IMG_SIZE = (34, 26)
PATH = './weights/trained.pth'
detect_face = dlib.get_frontal_face_detector()
predict_face = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
model = Net()
model.load_state_dict(torch.load(PATH))
model.eval()
n_count = 0

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

def predict_eye_state(eye_img):
    eye_img = cv2.resize(eye_img, dsize=IMG_SIZE)
    eye_input = eye_img.copy().reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32)
    eye_input = torch.from_numpy(eye_input)
    pred = model(eye_input)
    return torch.round(torch.sigmoid(pred)).item()

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, img_ori = cap.read()

    if not ret:
        break

    img_ori = cv2.resize(img_ori, dsize=(0, 0), fx=0.5, fy=0.5)
    img = img_ori.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detect_face(gray)

    for face in faces:
        shapes = predict_face(gray, face)
        shapes = face_utils.shape_to_np(shapes)
        eye_img_l, eye_rect_l = crop_eye(gray, eye_points=shapes[36:42])
        eye_img_r, eye_rect_r = crop_eye(gray, eye_points=shapes[42:48])
        eye_state_l = predict_eye_state(eye_img_l)
        eye_state_r = predict_eye_state(eye_img_r)

        if eye_state_l == 0.0 and eye_state_r == 0.0:
            n_count += 1
        else:
            n_count = 0

        if n_count > 100:
            cv2.putText(img, "Sleeping", (120, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('result', img)
    
    if cv2.waitKey(1) == ord('x'):
        break

cap.release()
cv2.destroyAllWindows()
