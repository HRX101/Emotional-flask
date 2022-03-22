
import cv2
from flask import Flask,render_template,Response
import tensorflow as tf
import pyttsx3
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from random import shuffle
import numpy as np
import pyautogui
import pywhatkit as ph
import warnings
warnings.filterwarnings("ignore")

import os
import mediapipe as mp
app = Flask(__name__)
#def face_visu():
p12=""
convnet = input_data(shape=[50,50,1])
convnet = conv_2d(convnet, 32, 5, activation='relu')
    # 32 filters and stride=5 so that the filter will move 5 pixel or unit at a time
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 128, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)
convnet = fully_connected(convnet, 3, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate = 0.001, loss='categorical_crossentropy')
model = tflearn.DNN(convnet, tensorboard_verbose=1)
model.load("model.tflearn")

cascade="haarcascade_frontalface_default.xml"
def say(audio):
    engine=pyttsx3.init()
        #engine.say("hi Hrithik it's me ")
    voices=engine.getProperty('voices')
    engine.setProperty('voice',voices[1].id)
    vrate=200
    engine.setProperty('rate',vrate)
    engine.say(audio)
    engine.runAndWait()


face_detect=cv2.CascadeClassifier(cascade)
cap=cv2.VideoCapture(0)
my_label=""
l=""

l=""
pas=False
p1=""
s12=1
mpDraw=mp.solutions.drawing_utils
def face_visu():
    p=0
    p9=0
    while True:
        p+=1
        rate,frame=cap.read()
        frame=cv2.flip(frame,1)
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        face=face_detect.detectMultiScale(gray,1.1,4)

        for (x,y,w,h) in face:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            #cv2.rectangle(frame, (200,40),(450,350),(255, 0, 255), 2)
            pr=frame
            roi = gray[y:y+h,x:x+w]
            #roi = cv2.resize(roi, (200, 200))
    
            # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20,20))
                #detected_face= cv2.morphologyEx(detected_face, cv2.MORPH_CLOSE, kernel)
            detected_face = cv2.resize( roi,(50,50))
            detected_face = np.array(detected_face).reshape((-1,50,50,1))
            
        # label = np.argmax(model.predict(detected_face))
            conf=model.predict(detected_face)[0]
            idx=np.argmax(conf)
                
            confiedence="{:.2f}%".format(conf[idx]*100)
            model_out=model.predict(detected_face)
            #print(model_out)
                
            if np.argmax(model_out) == 0:
                my_label = 'happy'
                p9+=1
            elif np.argmax(model_out)==1:
                my_label="sad"
                """
                if s12!=None:
                    s12=ph.playonyt("happy day")
                """
                    
            elif np.argmax(model_out)==2:
                my_label="angry"
            else:
                    my_label = 'unkwnon'
                #print(my_label)
            cv2.putText(frame,my_label,(100,100),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),1)
            cv2.putText(frame,str(p),(120,120),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0))
            cv2.putText(frame,str(confiedence),(120,180),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0))
        """
    
        if(my_label=="happy" and p9>=1):
            say("you look happy today")
        """

        #cv2.imshow("hey",frame)
        rete, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
    

    
    
        
        
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/video_feed')
def video_feed():
    return Response(face_visu(), mimetype='multipart/x-mixed-replace; boundary=frame')
if __name__=='__main__':
    app.run(debug=True)