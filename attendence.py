import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path='D:\\internship\\attendence using ai\\images'
images=[]
classNames=[]
mylist=os.listdir(path)
print(mylist)
for cls in mylist:
    curimg=cv2.imread(f'{path}/{cls}')
    images.append(curimg)
    classNames.append(os.path.splitext(cls)[0])
print(classNames)

def findencod(images):
    encodelist=[]
    for img in images:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode=face_recognition.face_encodings(img)[0]
        encodelist.append(encode)
    return encodelist

def markatten(name):
    with open('attendence.csv','r+') as f:
        mydatalist=f.readlines()
        namelist=[]
        #print(mydatalist)
        for line in mydatalist:
            entry=line.split(',')
            namelist.append(entry[0])
        if name not in namelist:
            now=datetime.now()
            dtString=now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')

markatten('joker')

encodelistknown=findencod(images)
print("encoding completed")

cam=cv2.VideoCapture(0)

while True:
    _,img=cam.read()
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    facescurframe=face_recognition.face_locations(imgS)
    encodecurframe= face_recognition.face_encodings(imgS,facescurframe)

    for encode,faceloc in zip(encodecurframe,facescurframe):
        matches=face_recognition.compare_faces(encodelistknown,encode)
        facedis=face_recognition.face_distance(encodelistknown,encode)
        #print(facedis)
        matchIndex = np.argmin(facedis)

        if matches[matchIndex]:
            name=classNames[matchIndex].upper()
            #print(name)
            y1,x2,y2,x1=faceloc
            y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_SCRIPT_COMPLEX,1,(255,255,255),2)
            markatten(name)



    cv2.imshow('webcam',img)
    key=cv2.waitKey(1) & 0xff

    if key==ord('q'):
        break

cam.release()
cv2.destroyAllWindows()