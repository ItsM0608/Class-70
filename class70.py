import cv2
from cvzone.HandTrackingModule import HandDetector
import pyautogui
import numpy as np
import math
import time
screensize=pyautogui.size()
print("what is screensize: ",screensize)
screenwidth=screensize[0]
screenheight=screensize[1]
#print(screenwidth,screenheight)

width=640
height=480
frameR=100

prevX=0
prevY=0

curX=0
curY=0

number=0

smoothening=1


video=cv2.VideoCapture(0)

detector= HandDetector(detectionCon=0.8)

while True:
    try:
        dummy,cameraFeedImage=video.read()
        cameraFeedImage_flipped=cv2.flip(cameraFeedImage, 1)
        handsDetector=detector.findHands(cameraFeedImage_flipped, flipType = False)
        hands=handsDetector[0]           #boolean value to check hand detected
        cameraFeedImage_flipped=handsDetector[1]   #1 index gets all landmark values

        if hands:
            hand1=hands[0]
            lmlist1=hand1["lmList"]
            handType1=hand1["type"]

            fingers=detector.fingersUp(hand1)
            if len(lmlist1)>0:
                indexFingerTipX=lmlist1[8][0]
                indexFingerTipY=lmlist1[8][1]

                if fingers[1] ==1 and fingers[2]==0:
                    x3= np.interp(indexFingerTipX, (frameR,width-frameR),(0,screenwidth))
                    y3= np.interp(indexFingerTipY, (frameR,height-frameR),(0,screenheight))

                    curX=prevX+(x3-prevX)/smoothening
                    curY=prevY+(x3-prevY)/smoothening


                    pyautogui.moveTo(curX,curY)

                    cv2.circle(cameraFeedImage_flipped,(indexFingerTipX,indexFingerTipY),15,(0,255,0),cv2.FILLED)
                    prevX=curX
                    prevY=curY

                if fingers[1] ==1 and fingers[2]==1:
                    distance= math.dist(lmlist1[8],lmlist1[12])

                    indexFingerTipX=lmlist1[8][0]
                    indexFingerTipY=lmlist1[8][1]
                    middleFingerTipX=lmlist1[12][0]
                    middleFingerTipY=lmlist1[12][1]


                    cx=(indexFingerTipX+middleFingerTipX)//2
                    cy=(indexFingerTipY+middleFingerTipY)//2

                    cv2.line(cameraFeedImage_flipped,(indexFingerTipX,indexFingerTipY),(middleFingerTipX,middleFingerTipY),
                            (255,0,255),2)
                    

                    if distance<20:
                        print(distance)
                        cv2.circle(cameraFeedImage_flipped,(cx,cy),15,(0,255,0),cv2.FILLED)
                        pyautogui.click()

                if fingers[0] == 1 and fingers[1]==0 and fingers[2]==0 and fingers[3]==0 and fingers[4]==0:
                    pyautogui.scroll(300)

                if fingers[0] == 0 and fingers[1]==1 and fingers[2]==1 and fingers[3]==1 and fingers[4]==1:
                    pyautogui.scroll(-300)

                if fingers[0] == 0 and fingers[1]==0 and fingers[2]==0 and fingers[3]==0 and fingers[4]==0:
                    screenshotpath=f"screenshots/screenshot_{number}.png"
                    pyautogui.screenshot(screenshotpath)
                    number+=1
                    print("Snapshot Saved")
                    time.sleep(1)

                    

                

    except Exception as e:
        print(e)

    cv2.imshow("handDetection_flip",cameraFeedImage_flipped)
    if cv2.waitKey(25) == 27:
        break

video.release()
cv2.destroyAllWindows()