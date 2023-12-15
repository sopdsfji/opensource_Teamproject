import youtube_dl
import pafy

url = 'https://www.youtube.com/watch?v=ipyzW38sHg0'
video = pafy.new(url)
best = video.getbest(preftype = 'mp4')

cap = cv2.VideoCapture(best.url)

frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

while True:
	retval, img = cap.read()
    if not retval:
    	break
    cv2.imshow("video", img)
    key = cv2.waitKey(25)
    if key == 27:
    	break

if cap.isOpened():
	cap.release()
cv2.destroyAllWindows()
    
