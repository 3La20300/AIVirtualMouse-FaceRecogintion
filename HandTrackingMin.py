import cv2
import mediapipe as mp
import time
import math

cap=cv2.VideoCapture(0)

mpHands=mp.solutions.hands
hands=mpHands.Hands() #obj called Hands
mpDraw=mp.solutions.drawing_utils
tipIDs=[4,8,12,16,20]

pTime = 0
cTime = 0

while True:
    success, img= cap.read()
    imgRGB=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results=hands.process(imgRGB)


    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
             for id, lm in enumerate(handLms.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                print(id, cx, cy)

             mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)


    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 255), 3)


    cv2.imshow("Image", img)
    cv2.waitKey(1)