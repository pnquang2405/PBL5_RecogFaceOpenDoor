import cv2
import time 
import os


cam = cv2.VideoCapture(1)
count = 0
name = input()
os.chdir('E:\\FaceRecog\\Dataset\\FaceData\\raw')
path = os.getcwd()
if (name in os.listdir(path)) == False:
    os.mkdir(name)
path += '\\' + name

while count < 30:
    OK, frame = cam.read()
    time.sleep(0.5)

    cut_faces = cv2.resize(frame, (500,500))
    path_1 = path + '\\face_{}.jpg'.format(count)
    cv2.imwrite(path_1, cut_faces)
    count += 1

    cv2.imshow('FRAME', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()