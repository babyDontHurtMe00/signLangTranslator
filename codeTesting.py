import mediapipe as mp
import cv2
import time
import csvLibrary as cl

# cl.info()

######################################################## Result CSV creation
def createResultCSV():
    cap = cv2.VideoCapture(0)


    mpHands = mp.solutions.hands
    hands = mpHands.Hands()
    mpDraw = mp.solutions.drawing_utils

    pTime = 0
    cTime = 0
    timerTime = time.time()
    output = []
    headers=[]
    partOutput=[]


    # Run for 05 seconds
    while time.time()<(timerTime+5):
    # while True:
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

            #interesting detail:
            # print(results.multi_handedness[1]) gives left, but its index property = 0
            # print(results.multi_handedness[0]) gives right, but its index property = 1

            # print(results.multi_handedness)
            for handSide in results.multi_handedness:
                # print(handSide.classification[0].index)
                halfList1.append(handSide.classification[0].index)


            for handLms in results.multi_hand_landmarks:
                #print(handLms.landmark)
                qtrList=[]
                #this area has both hands seperately
                for id,lm in enumerate(handLms.landmark):
                    mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
                    #print(id,lm) #this gives position (as a fraction of img) for each pt
                    height,width,channel = img.shape #finding img parameters to multiply to lm to get coordinate 
                    cx,cy = int(lm.x*width), int(lm.y*height)
                    qtrList.append([id,cx,cy])
                    # if id==4 or id==3:
                        # print(id, cx,cy)
                        # qtrList.append([id,cx,cy])
                
                
                halfList2.append(qtrList)
            #this area is after a frame
            # print(halfList1)
            # print(halfList2)
            # print("\n\n\n")
        

            partOutput.append(dict(zip(halfList1,halfList2)))   
        output.append(partOutput)



        #displayingFP
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime= cTime
        cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_COMPLEX,2,(255,0,255),3)
        cv2.putText(img,str(int(5-(time.time()-timerTime))),(540,70),cv2.FONT_HERSHEY_COMPLEX,2,(255,0,255),3)

        cv2.imshow("Image",img)
        cv2.waitKey(1)
        # time.sleep(2)


    #creating headers list for csv
    from decimal import Decimal
    for i in range(2):
        for j in range(21):
            # print(j)
            headers.append(round(i+Decimal(j/100),2))


    #removing empty lists from output
    i=0
    while (i<len(output)):
        if output[i] != []:
            # print("\n\nOUTPUT: ",output[i])
            i+=1
        else:
            output.pop(i)
            # print(str(i)+"/"+str(len(output)))

    finalOutput = []
    #converting output to the perfect dictionary wali list
    for i in range(len(partOutput)):
        # print(output[i].keys())
        tempDict={}
        for j in partOutput[i].keys():
            for k in partOutput[i][j]:
                # print(round(j+Decimal(k[0]/100),2),k[1:])
                tempDict[round(j+Decimal(k[0]/100),2)] = k[1:]

        #creating dict of one frame and appending it here            
        finalOutput.append(tempDict)


    #printing headers, output        
    # print("HEADERS:",headers)
    # print("OUTPUT:",finalOutput)
    print("\n\nFrames Captured =",len(finalOutput))
    # left is 0, right is 1

    cl.dwrite(headers,finalOutput,"./currentOutput.csv")

############################################################### Artifacts loading
import joblib
import json

__class_name_to_number = {}
__class_number_to_name = {}

__model = None


def class_number_to_name(class_num):
    return __class_number_to_name[class_num]

def load_saved_artifacts():
    # print("loading saved artifacts...start")
    global __class_name_to_number
    global __class_number_to_name

    with open("./artifacts/class_dictionary.json", "r") as f:
        __class_name_to_number = json.load(f)
        __class_number_to_name = {v:k for k,v in __class_name_to_number.items()}

    global __model
    if __model is None:
        with open('./artifacts/saved_model.pkl', 'rb') as f:
            __model = joblib.load(f)
    print("Artifacts Loaded")
