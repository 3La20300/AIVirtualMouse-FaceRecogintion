import cv2
import mediapipe as mp
import time
import math
import autopy
import numpy as np


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

            if draw:
                cv2.rectangle(img, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20),
                              (0, 255, 0), 2)

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



##########################
wCam, hCam = 640, 480
frameR=100
smoothening=7
plocX, plocY=0, 0 #prevLocation
cLocX, cLocY=0, 0 #currentLocation
#########################

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
detector = handDetector(maxHands=1)
wScr, hScr = autopy.screen.size()
print(wScr, hScr)
while True:
    # 1. Find hand Landmarks
    success, img = cap.read()
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)
    # 2. Get the tip of the index and middle fingers
    x1= 0
    y1=0
    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]
    # 3. Check which fingers are up
    fingers = detector.fingersUp()
    # print(fingers)
    cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR),
                  (255, 0, 255), 2)
    # (frameR, frameR), (wCam-frameR, hCam-frameR) start and end points of our rec
    # 5. Convert Coordinates
    x3 = np.interp(x1, (frameR, wCam-frameR), (0, wScr)) # convert from the range of the rectangle to the range of screen when u reach the top of rectangle u reach the top of the screen
    y3 = np.interp(y1, (frameR, hCam-frameR), (0, hScr)) # same as up
    #Smotheen Values
    # 6. Smoothen Values
    clocX = plocX + (x3 - plocX) / smoothening
    clocY = plocY + (y3 - plocY) / smoothening

    plocX, plocY = clocX, clocY
    #7. Move Mouse
    try:
        autopy.mouse.move(wScr - clocX, clocY)
        cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
    except:
        autopy.mouse.move(clocX, clocY)

    #8. Both Index and middle fingers are up : Clicking Mode
    if fingers[1]== 1 and fingers[2]==1:
        #9. Find Distance between fingers
        length, img, lineInfo = detector.findDistance(8 ,12 ,img)
        #10. Click mouse if distance short
        try:
          if length<40:
           cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0,255,0),cv2.FILLED)
           autopy.mouse.click()
        except:
           cv2.circle(img, (x1,y1), 15, (0, 255, 0), cv2.FILLED)
           autopy.mouse.click()

    # if len(lmList) != 0:
    #      print(lmList[4])

    # pTime = cTime
    # cTime = time.time()
    # fps = 1 / (cTime - pTime)

    # cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
    #                 (255, 0, 255), 3)

    cv2.imshow("Mouse", img)
    cv2.waitKey(1)

