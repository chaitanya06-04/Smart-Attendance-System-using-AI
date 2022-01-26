import cv2
import numpy as np
import face_recognition
imgjoker=face_recognition.load_image_file('D:\\internship\\attendence using ai\\images\\batmanjokerheathledger_1.jpeg')
imgjoker=cv2.cvtColor(imgjoker,cv2.COLOR_BGR2RGB)
imgtest=face_recognition.load_image_file('D:\\internship\\attendence using ai\\images\\batmanjokerheathledger_2.jpeg')
imgtest=cv2.cvtColor(imgtest,cv2.COLOR_BGR2RGB)
faceloc=face_recognition.face_locations(imgjoker)[0]
encodejoker=face_recognition.face_encodings(imgjoker)[0]
cv2.rectangle(imgjoker,(faceloc[3],faceloc[0]),(faceloc[1],faceloc[2]),(255,0,255),2)

faceloctest=face_recognition.face_locations(imgtest)[0]
encodetest=face_recognition.face_encodings(imgtest)[0]
cv2.rectangle(imgtest,(faceloctest[3],faceloctest[0]),(faceloctest[1],faceloctest[2]),(255,0,255),2)

results=face_recognition.compare_faces([encodejoker],encodetest)
#print(results)
facedis=face_recognition.face_distance([encodejoker],encodetest)
#print(facedis)
cv2.putText(imgtest,f'{results}',(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)

cv2.imshow('joker',imgjoker)
cv2.imshow('joker test',imgtest)

cv2.waitKey(0)

