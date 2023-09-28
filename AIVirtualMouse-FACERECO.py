import mediapipe as mp
import time
import math
import autopy
import numpy as np
import cv2
import face_recognition
import cvzone
from datetime import datetime


#-----------------------------AIVirtualMouse------------------------------
class handDetector():
    def __init__(self, mode=False, modelComplexity=1, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplex = modelComplexity
        self.detectionCon = 0.5
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplex, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        # self.tipIDs = [4, 8, 12, 16, 20]

    def findHands(self, img, draw=True):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

        return img

    def findPosition(self, img, handNo=0, draw=True):
        xList = []
        yList = []
        bbox = []
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                # print(id, cx, cy)
                self.lmList.append([id, cx, cy])
                # if draw:
                #   cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
            xmin, xmax = min(xList), max(yList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax

            # if draw:
            #     cv2.rectangle(img, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20),
            #                   (0, 255, 0), 2)

        return self.lmList, bbox

    def fingersUp(self):
        fingers = []
        tipIds=[4 , 8 , 12 ,16 ,20]
        # Thumb
        try:
            if self.lmList[4][1] > self.lmList[4 - 1][1]:
             fingers.append(1)
            else:
                fingers.append(0)
        except:
            fingers.append(0)
        for id in range(1, 5):
           try:
            if self.lmList[tipIds[id]][2] < self.lmList[tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
           except:
            fingers.append(0)
        # totalFingers = fingers.count(1)

        return fingers

    def findDistance(self, p1, p2, img, draw=True, r=15, t=3):
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)
        length = math.hypot(x2 - x1, y2 - y1)

        return length, img, [x1, y1, x2, y2, cx, cy]
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------

#------------AIVirtualMouse---------
##########
detector = handDetector(maxHands=1)
wScr, hScr = autopy.screen.size()
print(wScr, hScr)
#########
##########################
matchTrue=0
wCam, hCam = 640, 480
frameR=100
smoothening=7
plocX, plocY=0, 0 #prevLocation
cLocX, cLocY=0, 0 #currentLocation
#########################
#------------------------------------


cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
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
                matchTrue=1
                break

            elif matches[0]==False :
                cv2.putText(img, "NO MATCH!!", (270, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            cv2.imshow("AIVirtualMouse_FaceRecognition", img)
            cv2.waitKey(1)

    if matchTrue==1:
        break

if matchTrue==1:
 import AIVirtualMouse
