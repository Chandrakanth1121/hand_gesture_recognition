import cv2
import numpy as np
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.models import load_model
from picamera.array import PiRGBArray
import time
from picamera import Picamera

model=load_model('gestures.h5')
model.summary()

camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640, 480))

time.sleep(0.1)

font = cv2.FONT_HERSHEY_SIMPLEX

for slide in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
	img = slide.array
	cv2.imshow("Frame",img)	
	frame=cv2.flip(img,1)
        cv2.rectangle(frame, (300,0), (600,400), (225, 225, 225), 3)
        roi=frame[0:400,300:600]
        roi= cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        roi = cv2.GaussianBlur(roi, (5, 5), cv2.BORDER_DEFAULT)

        cv2.imwrite('predict.jpg',roi)

        img_pred = image.load_img('predict.jpg', target_size=(150,150,1))
        img_pred = image.img_to_array(img_pred)
        img_pred = np.expand_dims(img_pred, axis=0)

        rslt = model.predict(img_pred)
        arr=np.rint(rslt)
        if(arr[0,0]==1):
            print("down")
            cv2.putText(roi, 'down',(50,50), font,2, (0,255,0), 3, cv2.LINE_AA)
        elif(arr[0,1]==1):
            print("fist")
            cv2.putText(roi, 'fist', (50,50), font, 2, (0, 255, 0), 3, cv2.LINE_AA)
        elif (arr[0, 2] == 1):
            print("left")
            cv2.putText(roi, 'left', (50, 50), font, 2, (0, 255, 0), 3, cv2.LINE_AA)
        elif (arr[0, 3] == 1):
            print("open")
            cv2.putText(roi, 'open', (50, 50), font, 2, (0, 255, 0), 3, cv2.LINE_AA)
        elif (arr[0, 4] == 1):
            print("right")
            cv2.putText(roi, 'right', (50, 50), font, 2, (0, 255, 0), 3, cv2.LINE_AA)
        else:
            print('up')
            cv2.putText(roi, 'up', (50, 50), font, 2, (0, 255, 0), 3, cv2.LINE_AA)
        #cv2.imshow('', frame)
        cv2.imshow('roi',roi)
        cv2.imshow('sk', frame)
	
	rawCapture.truncate(0)
	
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
 
cap.release()
cv2.destroyAllWindows()

