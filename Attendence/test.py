from sklearn.neighbors import KNeighborsClassifier
import cv2
import numpy as np
import pickle
import os
import time
from datetime import datetime
import csv

from win32com.client import Dispatch
def speak(str1):
      speak = Dispatch("SAPI.SpVoice")
      speak.Speak(str1)
video = cv2.VideoCapture(0)
facedtect = cv2.CascadeClassifier('Data\haarcascade_frontalface_default.xml')

with open('data/names.pkl','rb') as f:
        LABELS = pickle.load(f)

with open('data/faces_data.pkl','rb') as f:
        FACES= pickle.load(f)
knn= KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES,LABELS)

COL_NAME = ['NAME' ,'TIME']

while True:
    ret , frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedtect.detectMultiScale(gray, 1.3,5)
    for(x,y,w,h) in faces:
        crop_img = frame[y:y+h , x:x+w,:]
        resize_img = cv2.resize(crop_img,(50,50)).flatten().reshape(1,-1)
        output = knn.predict(resize_img)
        ts=time.time()
        date=datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        timestamp = datetime.fromtimestamp(ts).strftime("%H-%M-%S")
        exists=os.path.isfile("Attendence/Attendence_" + date+ ".csv")

        cv2.putText(frame, str(output[0]), (x,y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 2)  
        attendence = [str(output[0]),str(timestamp)]

    cv2.imshow("frame",frame)
    k = cv2.waitKey(1)
    if k==ord('o'):
          speak("Attendence Taken " )
          time.sleep(5)
          if exists:
                with open("Attendence/Attendence_" + date+ ".csv","+a") as csvfile:
                      writer = csv.writer(csvfile)
                      
                      writer.writerow(attendence)
                csvfile.close()
          else :
                with open("Attendence/Attendence_" + date+ ".csv","+a") as csvfile:
                      writer = csv.writer(csvfile)
                      writer.writerow(COL_NAME)
                      writer.writerow(attendence)
                csvfile.close()
    if k ==ord('q') :
        break
video.release()
cv2.destroyAllWindows()

