import cv2
import numpy as np
from cv2 import waitKey
import mediapipe as mp
import pyautogui


cap=cv2.VideoCapture(0)
face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
#eye_cascade=cv2.CascadeClassifier('')
hid=0
while True:
    ret,frame=cap.read()
    
   
    frame=cv2.flip(frame,1)
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    face=face_cascade.detectMultiScale(gray,1.1,4)
    print(face)
    for (x,y,w,h) in face:
        hid+=1
        roi=frame[y:y+h,x:x+w]
        roi1=gray[y:y+h,x:x+w]
        cv2.putText(frame,str(hid),(x,y),cv2.FONT_HERSHEY_TRIPLEX,1,(255,0,0),2)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.imwrite("data/"+'angry.'+str(hid)+'.png',roi1)
    cv2.imshow("face",frame)
    

            
    


    if cv2.waitKey(1)==ord("a") or hid==502:
        break

cap.release()
cv2.destroyAllWindows()

