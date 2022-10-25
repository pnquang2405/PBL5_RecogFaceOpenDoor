from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from asyncore import loop
from cv2 import VideoCapture

import tensorflow as tf
from imutils.video import VideoStream
import requests
from datetime import datetime
from re import T
import argparse
import facenet
import imutils
import os
import sys
import math
import pickle
import align.detect_face
import numpy as np
import cv2
import collections
from sklearn.svm import SVC
# import RPI.GPIO as GPIO
import pyrebase

import time as time
 
# GPIO.setmode(GPIO.BCM)
# GPIO.setup(18, GPIO.OUT)
 

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
storage = firebase.storage()
    

# with open('labels', 'rb') as f:
# 	dict = pickle.load(f)
# 	f.close()

cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


def getNumberUser():
        count=0
        all_users = database.child("User").get()
        if all_users==None:
                return count
        for users in all_users.each():
                count+=1
        return count

font = cv2.FONT_HERSHEY_SIMPLEX
count1=getNumberUser()+1
print(count1)

def SentData(name,frame,count1, status=True):
        name1=name+str(count1)+".jpg"
        now=datetime.now()
        dt=now.strftime("%d/%m/%Y :%H:%M")
        print(dt)
        cv2.imwrite(name1,frame)
        auth = firebase.auth()
        email = "thanhthan2k1@gmail.com"
        password = "thanhthan"
        user = auth.sign_in_with_email_and_password(email, password)
        storage.child(name1).put(name1)
        

        imageUrl = storage.child(name1).get_url(user['idToken'])
                #print(imageUrl)
        data={"avatar":imageUrl,"name":name,"time":dt}
        #database.set(data)
        database.child("User").child("User"+str(count1))
                #user.setValue(data);
        database.set(data)
        if not status == False:
            database.child("Door")
            database.set("true")
        print("da gui anh"+str(count1))
        os.remove(name1)
        count1 = count1+1
                #sleep(1)

def closeDoor():
    database.child("Door")
    database.set("false")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help='Path of the video you want to test on.', default=0)
    args = parser.parse_args()

    MINSIZE = 20
    THRESHOLD = [0.6, 0.7, 0.7]
    FACTOR = 0.709
    IMAGE_SIZE = 182
    INPUT_IMAGE_SIZE = 160
    CLASSIFIER_PATH = 'models/facemodel.pkl' 
    VIDEO_PATH = args.path
    FACENET_MODEL_PATH = 'models/20180402-114759.pb'

    # Load The Custom Classifier
    with open(CLASSIFIER_PATH, 'rb') as file:
        model, class_names = pickle.load(file)
    print("Custom Classifier, Successfully loaded")

    with tf.Graph().as_default():

        gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.6)
        sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options, log_device_placement=False))

        with sess.as_default():

            # Load the model
            print('Loading feature extraction model')
            facenet.load_model(FACENET_MODEL_PATH)  

            # Get input and output tensors
            images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("phase_train:0")
            # embedding_size = embeddings.get_shape()[1]

            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, "src/align")

            # people_detected = set()
            person_detected = collections.Counter()

            # cap  = VideoStream().start()
            
            cap= cv2.VideoCapture("http://192.168.43.200:5000/video_feed")
            #cap = cv2.VideoCapture(1)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 15)
            
            #cap.set(cv2.CAP_PROP_BUFFERSIZE,4)


            name = "unknown"
            while (True):
                data = database.child("Door").get()
                if(data.val() == True):
                    x = requests.get('http://192.168.43.200:5000/unlock')
                    time.sleep(5)
                    x = requests.get('http://192.168.43.200:5000/lock')
                    closeDoor()
                    continue

                frame = cap.read()
                frame = imutils.resize(np.array(frame[1]), width=600)
                frame = cv2.flip(frame, 1)

                bounding_boxes, _ = align.detect_face.detect_face(frame, MINSIZE, pnet, rnet, onet, THRESHOLD, FACTOR)

                faces_found = bounding_boxes.shape[0]
                try:
                    if faces_found > 1:
                        cv2.putText(frame, "Only one face", (0, 100), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                    1, (255, 255, 255), thickness=1, lineType=2)
                    elif faces_found > 0:
                        det = bounding_boxes[:, 0:4]
                        bb = np.zeros((faces_found, 4), dtype=np.int32)
                        for i in range(faces_found):
                            bb[i][0] = det[i][0]
                            bb[i][1] = det[i][1]
                            bb[i][2] = det[i][2]
                            bb[i][3] = det[i][3]
                            #print(bb[i][3]-bb[i][1])
                            #print(frame.shape[0])
                            #print((bb[i][3]-bb[i][1])/frame.shape[0])
                            if (bb[i][3]-bb[i][1])/frame.shape[0]>0.25: 
                                cropped = frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :]
                                scaled = cv2.resize(cropped, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE),
                                                    interpolation=cv2.INTER_CUBIC)
                                scaled = facenet.prewhiten(scaled)
                                scaled_reshape = scaled.reshape(-1, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3)
                                feed_dict = {images_placeholder: scaled_reshape, phase_train_placeholder: False}
                                emb_array = sess.run(embeddings, feed_dict=feed_dict)

                                predictions = model.predict_proba(emb_array)
                                best_class_indices = np.argmax(predictions, axis=1)
                                best_class_probabilities = predictions[
                                    np.arange(len(best_class_indices)), best_class_indices]
                                best_name = class_names[best_class_indices[0]]
                                # print("Name: {}, Probability: {}".format(best_name, best_class_probabilities))
                                print(best_class_probabilities)
                                print(best_name)
                                if best_class_probabilities > 0.8:
                                    print(best_class_probabilities)
                                    print(class_names[best_class_indices[0]])
                                    x = requests.get('http://192.168.43.200:5000/unlock')
                                    SentData(class_names[best_class_indices[0]],frame,getNumberUser()+1)
                                    for i in range(0,15*10-1):
                                        cap.grab()
                                    x = requests.get('http://192.168.43.200:5000/lock')
                                    closeDoor()
                                    cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)
                                    text_x = bb[i][0]
                                    text_y = bb[i][3] + 20

                                    name = class_names[best_class_indices[0]]
                                    cv2.putText(frame, name, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                1, (255, 255, 255), thickness=1, lineType=2)
                                    cv2.putText(frame, str(round(best_class_probabilities[0], 3)), (text_x, text_y + 17),
                                                cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                1, (255, 255, 255), thickness=1, lineType=2)
                                    person_detected[best_name] += 1
                                    print(name)
                                    print("da vao day")
                                else:
                                    SentData("Unknown",frame,count1, False)
                                    continue

                except:
                    pass

                cv2.imshow('Face Recognition', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            #cv2.destroyWindow()
main()