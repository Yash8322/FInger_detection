import cv2
import time
import os
import Hand_Track_Finger as htf
WCAM, HCAM = 1920, 1080
out = cv2.VideoCapture(1)
out.set(3, WCAM)
out.set(4, HCAM)
folderPath = "FingerImages"
myList = os.listdir(folderPath)
print(myList)
overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    #print(f'{folderPath}/{imPath}')
    overlayList.append(image)
print(len(overlayList))
pTime = 0
detector = htf.handDetector(detectionCon=0.75)
tipIds = [4, 8, 12, 16, 20]
while True:
    success, img = out.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    #print(lmList)
    if len(lmList) != 0:
        fingers = []
        if lmList[tipIds[0]][1] > lmList[tipIds[0]-1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        for id in range(1, 5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        #print(fingers)
        totalFingers = fingers.count(1)
        print(totalFingers)
        img[0:300, 0:300] = overlayList[totalFingers - 1]
        cv2.rectangle(img, (40, 655), (250, 925), (0, 255, 255), cv2.FILLED)
        cv2.putText(img, str(totalFingers), (90, 790), cv2.FONT_HERSHEY_PLAIN, 10, (255, 255, 255), 20)
    cTime = time.time()
    FPS = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(FPS)}', (600, 80), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 0), 3)
    if cv2.waitKey(1) == ord('8'):
        break
    cv2.imshow("Image", img)
