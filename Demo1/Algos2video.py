import cv2
import math
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator


#frame = cv2.imread(fpath)
mode = "display"
padding=20


model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))


model.load_weights('model.h5')

# prevents openCL usage and unnecessary logging messages
cv2.ocl.setUseOpenCL(False)

# dictionary which assigns each label an emotion (alphabetical order)
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

faceProto="opencv_face_detector.pbtxt"
faceModel="opencv_face_detector_uint8.pb"
ageProto="age_deploy.prototxt"
ageModel="age_net.caffemodel"
genderProto="gender_deploy.prototxt"
genderModel="gender_net.caffemodel"

MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList=['Male','Female']

faceNet=cv2.dnn.readNet(faceModel,faceProto)
ageNet=cv2.dnn.readNet(ageModel,ageProto)
genderNet=cv2.dnn.readNet(genderModel,genderProto)




cap=cv2.VideoCapture(0)
fourcc=cv2.VideoWriter_fourcc(*'XVID')
out=cv2.VideoWriter('output.avi',fourcc,25.0,(640,480))    ####### à changer peut etre



###########################################################

def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn=frame.copy()
    frameHeight=frameOpencvDnn.shape[0]
    frameWidth=frameOpencvDnn.shape[1]
    blob=cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)
    
    hw=max(frameHeight,frameHeight)
    net.setInput(blob)
    detections=net.forward()
    faceBoxes=[]
    
    for i in range(detections.shape[2]):
        confidence=detections[0,0,i,2]
        if confidence>conf_threshold:
            x1=int(detections[0,0,i,3]*frameWidth)
            y1=int(detections[0,0,i,4]*frameHeight)
            x2=int(detections[0,0,i,5]*frameWidth)
            y2=int(detections[0,0,i,6]*frameHeight)
            if (x1<hw) and (x2<hw) and (y1<hw) and (y2<hw):
                faceBoxes.append([x1,y1,x2,y2])
                cv2.rectangle(frameOpencvDnn, (x1,y1), (x2,y2), (0,255,0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn,faceBoxes
#############################################################"
def Emotion(face):
        

        framegray= cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(framegray, (48, 48)), -1), 0)

        prediction = model.predict(cropped_img)

        maxindex = int(np.argmax(prediction))
        emotion=emotion_dict[maxindex]
        

                
        return emotion
###########################################################################




while (cap.isOpened()):
  
    ret,frame=cap.read()
    if (ret==True):
        frame=cv2.flip(frame,1)
        #out.write(frame)
        #cv2.imshow('frame',frame)
        resultImg,faceBoxes=highlightFace(faceNet,frame)
        if not faceBoxes:
            print("No face detected")

        for faceBox in faceBoxes:
                print(faceBox)
                

                face=frame[max(0,faceBox[1]-padding):
               min(faceBox[3]+padding,frame.shape[0]-1),max(0,faceBox[0]-padding)
               :min(faceBox[2]+padding, frame.shape[1]-1)]

                blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
                genderNet.setInput(blob)
                genderPreds=genderNet.forward()
                gender=genderList[genderPreds[0].argmax()]
                print(f'Gender: {gender}')

                ageNet.setInput(blob)
                agePreds=ageNet.forward()
                age=ageList[agePreds[0].argmax()]
                print(f'Age: {age[1:-1]} years')
                
                
                emotion=Emotion(face)
                
                
                

                cv2.putText(resultImg, f'{gender}, {age},{emotion}', (faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
                cv2.imshow("Detecting", resultImg)
                out.write(resultImg)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):

        
            break
            
    else:
        break
        
 
            
cap.release()        
out.release()   
cv2.destroyAllWindows()