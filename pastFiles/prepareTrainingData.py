import mediapipe as mp
import cv2
import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0
timerTime = time.time()
output = []

# Run for 10 seconds
while time.time()<(timerTime+10):
    success, img= cap.read()
    img=cv2.flip(img,1)
    # cv2.imshow("showing live feed",success)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    

    halfList1 = []
    halfList2 = []
    qtrList = []

    if (results.multi_hand_landmarks):
        # print("next frame") 


        for handSide in results.multi_handedness:
            halfList1.append(handSide.classification[0].label)
            # if handSide.classification[0].label == "Right":
            #     halfList1.append("Left")
            #     # print("Left")
            # else:
            #     halfList1.append("Right")
            #     # print("Right")

        for handLms in results.multi_hand_landmarks:
            #print(handLms.landmark)
            qtrList=[]
            for id,lm in enumerate(handLms.landmark):
                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
                #print(id,lm) #this gives position (as a fraction of img) for each pt
                height,width,channel = img.shape #finding img parameters to multiply to lm to get coordinate 
                cx,cy = int(lm.x*width), int(lm.y*height)
                qtrList.append({id: {"x":cx,"y":cy}})
                # print(id, cx,cy)
            halfList2.append(qtrList)
        output.append(dict(zip(halfList1,halfList2)))   



    #displayingFP
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime= cTime
    cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_COMPLEX,2,(255,0,255),3)
    cv2.putText(img,str(int(20-(time.time()-timerTime))),(540,70),cv2.FONT_HERSHEY_COMPLEX,2,(255,0,255),3)

    cv2.imshow("Image",img)
    cv2.waitKey(1)
    # time.sleep(2)

print(output)