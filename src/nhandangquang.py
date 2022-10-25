from datetime import datetime
from re import T
import string
import cv2
from PIL import Image
import numpy as np 
import pickle
from time import sleep
import pyrebase;
import os

firebaseConfig = {
    'apiKey': "AIzaSyAwO4hBvYnKCPp5_0tWw4BeK1eEgH71IeY",
    'authDomain': "mypbl5-88973.firebaseapp.com",
    'databaseURL': "https://mypbl5-88973-default-rtdb.firebaseio.com",
    'projectId': "mypbl5-88973",
    'storageBucket': "mypbl5-88973.appspot.com",
    'messagingSenderId': "233060726166",
    'appId': "1:233060726166:web:94afbe5b70a70c534c99b7",
    'measurementId': "G-ZC4HB3ZT63"   
    }
firebase = pyrebase.initialize_app(firebaseConfig)
database=firebase.database()
storage = firebase.storage();
    

with open('labels', 'rb') as f:
	dict = pickle.load(f)
	f.close()

cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

auth = firebase.auth()
email = "thanhthan2k1@gmail.com"
password = "thanhthan"
user = auth.sign_in_with_email_and_password(email, password)


faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")
def getNumberUser():
        count=0;
        all_users = database.child("User").get()
        for users in all_users.each():
                count+=1
        return count
   
font = cv2.FONT_HERSHEY_SIMPLEX
count=getNumberUser()+1
print(count)
def SentData(name,frame,):

        name1=str(count)+".jpg"
        now=datetime.now()
        dt=now.strftime("%d/%m/%Y :%H:%M")
                #print(dt);
        cv2.imwrite(name1,frame)
        storage.child(name1).put(name1)
                #database=firebase.database()
                
                #user = database.child("Users");
                
                #database.child("User"+str(count))
                
        # Enter your user account details 
        

        imageUrl = storage.child(name1).get_url(user['idToken'])
                #print(imageUrl)
        data={"avatar":imageUrl,"name":name,"time":dt}
        database.set(data)
                #database.child("User").child("User"+str(count));
                #user.setValue(data);
                #database.set(data)
        print("da gui anh")
        os.remove(name1)
        count=count+1
                #sleep(1)
       
        
        name1="UnKnown"+str(count)+".jpg"
        now=datetime.now()
        dt=now.strftime("%d/%m/%Y :%H:%M")
                #print(dt);
        cv2.imwrite(name1,frame)
        storage.child(name1).put(name1)
                
        database=firebase.database()
                #user = database.child("Users");     
                #database.child("User"+str(count))
        auth = firebase.auth()    
        # Enter your user account details 
        imageUrl = storage.child(name1).get_url(user['idToken'])
                #print(imageUrl)
        data={"avatar":imageUrl,"name":name1,"time":dt}
        database.child("User").child("User"+str(count))
        database.set(data)
        print("da gui anh")
        os.remove(name1)
        count=count+1
while True:
    Ok,frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor = 1.5, minNeighbors = 5)
    
    for (x, y, w, h) in faces:
        roiGray = gray[y:y+h, x:x+w]
        id_, conf = recognizer.predict(roiGray)
        for name, value in dict.items():
            if value == id_: ## nếu nhận dạng đc
                print(name)
                SentData(name,frame)
                break
        
        
        if conf <= 70:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, name + str(conf), (x, y), font, 2, (0, 0 ,255), 2,cv2.LINE_AA)
    cv2.imshow('frame', frame)
    
    key = cv2.waitKey(1)
    if key == 27:
        break
cam.release()
cv2.destroyAllWindows()