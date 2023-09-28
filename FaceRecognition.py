import os
import pickle
import numpy as np
import cv2
import face_recognition
import cvzone
from datetime import datetime


cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
ref_img=cv2.imread("MyImg.jpg")
RGB_ref_img= cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
encodeList=[]
encode = face_recognition.face_encodings(RGB_ref_img)[0]
encodeList.append(encode)



while True:
    success, img = cap.read()

    # imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    faceCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)


    for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):

            matches = face_recognition.compare_faces(encodeList, encodeFace)
            faceDis = face_recognition.face_distance(encodeList, encodeFace)
            cv2.rectangle(img, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (0, 0, 255), 2)
            # cv2.putText(img, "NO MATCH!!", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

            # matchIndex = np.argmin(faceDis)
            # print("Match Index", matchIndex)

            if matches[0]:
                # print("Known Face Detected")
                # print(studentIds[matchIndex])
                # y1, x2, y2, x1 = faceLoc
                # y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                # bbox = 55 + x1, 162 + y1, x2 - x1, y2 - y1
                cv2.rectangle(img, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (0, 255, 0), 2)
                cv2.putText(img, "MATCH!!", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            elif matches[0]==False :
                cv2.putText(img, "NO MATCH!!", (270, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)



    # cv2.imshow("Webcam", img)
    cv2.imshow("Face Recognition", img)
    cv2.waitKey(1)