import numpy as np
import mediapipe as mp
import cv2

cap=cv2.VideoCapture(0) 

name=input("Enter the name of the data: ")


holistic=mp.solutions.holistic
hands=mp.solutions.hands
holis=holistic.Holistic()
drawing=mp.solutions.drawing_utils

x=[]
datasize=0

while True:
    lst=[]
    _, frm=cap.read()
    # frm = cv2.flip(frm, 1)
    res=holis.process(cv2.cvtColor(frm,cv2.COLOR_BGR2RGB))
    cv2.imshow("window",frm)

    if res.face_landmarks:
        for i in res.face_landmarks.landmark:
            lst.append(i.x - res.face_landmarks.landmark[1].x)
            lst.append(i.y - res.face_landmarks.landmark[1].y)
    
    
        if res.left_hand_landmarks:
            for i in res.left_hand_landmarks.landmark:
                lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
                lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
        else:
            for i in range(42):
                lst.append(0.0)


        if res.right_hand_landmarks:
            for i in res.right_hand_landmarks.landmark:
                lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
                lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
        else:
            for i in range(42):
                lst.append(0.0)

        x.append(lst)
        datasize=datasize+1




        

    drawing.draw_landmarks(frm, res.face_landmarks,holistic.FACE_CONNECTIONS)
    drawing.draw_landmarks(frm, res.left_hand_landmarks,hands.HAND_CONNECTIONS)
    drawing.draw_landmarks(frm, res.right_hand_landmarks,hands.HAND_CONNECTIONS)

    cv2.putText(frm,str(datasize),(50,50), cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

    
    if cv2.waitKey(1)==27 or datasize>99 :
        cv2.destroyAllWindows()
        cap.release()
        break

np.save("{name}.npy",np.array(x))
print(np.array(x).shape)

