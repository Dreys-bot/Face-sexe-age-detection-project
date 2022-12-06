######### Detection de visage par l'algo OpenCV2, statistique 


import os
import cv2
import numpy as np
import time 



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


Input_directory=os.path.dirname(os.path.abspath(__file__))+r"/imgs/Input/"
Output_directory=os.path.dirname(os.path.abspath(__file__))+r"/imgs/Output/"

FileProb_directory=os.path.dirname(os.path.abspath(__file__))+r"/imgs/Output_Problem_File/"



########################################################
def Detect_face(fpath, conf_threshold=0.7): 
    
    frame = cv2.imread(fpath)
    framebgr = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    net=cv2.dnn.readNet(faceModel,faceProto)
    
 
    
    frameOpencvDnn2=framebgr.copy()
    frameOpencvDnn=frame.copy()
    frameHeight=frameOpencvDnn.shape[0]
    frameWidth=frameOpencvDnn.shape[1]
    blob=cv2.dnn.blobFromImage(frameOpencvDnn2, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections=net.forward()
    
    hw1=frame.shape[0]
    hw2=frame.shape[1]
    hw=max(hw1,hw2)
    
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
    return faceBoxes,frameOpencvDnn
  
########################################################
def Age_Gender(fpath,faces,resultImg):
    
            padding=20
            ageFile=[]
            genreFile=[]
       
            frame = cv2.imread(fpath)
           
            
            framebgr = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
          
            for faceBox in faces:
                

                face=frame[max(0,faceBox[1]-padding):
                           min(faceBox[3]+padding,frame.shape[0]-1),max(0,faceBox[0]-padding)
                           :min(faceBox[2]+padding, frame.shape[1]-1)]
               
                blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
                genderNet.setInput(blob)
                genderPreds=genderNet.forward()
                gender=genderList[genderPreds[0].argmax()]
                print(gender)
                genreFile.append(gender)

                ageNet.setInput(blob)
                agePreds=ageNet.forward()
                age=ageList[min(agePreds[0].argmax()+1,7)]
                print(age)
                ageFile.append(age)



                cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 5, (0,255,255), 8, cv2.LINE_AA) #2
            return   0,resultImg,genreFile,ageFile
  


 #################################################################
def Emotion(fpath,resultImg,facesV):
        
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Dropout, Flatten
        from tensorflow.keras.layers import Conv2D
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.layers import MaxPooling2D
        from tensorflow.keras.preprocessing.image import ImageDataGenerator

       
        frame = cv2.imread(fpath)
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


  

        framegray= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        emotionFile=[]
        for (x1, y1, x2, y2) in facesV:
            
            x=max(x1,1)
            y=max(y1,1)
            w=x2-x1
            h=y2-y1
       
            
            roi_gray = framegray[y:y + h, x:x + w]
            
            if h<48 and w<48:   # si frame était petit
                scale_percent = 150 # percent of original size
                width = int(roi_gray.shape[1] * scale_percent / 100)
                height = int(roi_gray.shape[0] * scale_percent / 100)
                dim = (width, height)

    # resize image
                
                roi_gray = cv2.resize(roi_gray, dim, interpolation = cv2.INTER_AREA)
            try:
                cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)

                prediction = model.predict(cropped_img)

                maxindex = int(np.argmax(prediction))
                print(emotion_dict[maxindex])
                cv2.putText(resultImg, emotion_dict[maxindex], (x2-w, y2+10),cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 255), 8, cv2.LINE_AA)
                

                emotionFile.append(emotion_dict[maxindex])
            except:
                
                 emotionFile.append(" ")
                
        return 0,resultImg,emotionFile

#########################################
def modification_date_time(filename):
    
    created = os.path.getmtime(filename)
    year,month,day,hour,minute,second=time.localtime(created)[:-3]
    
    dateT="%02d/%02d/%d"%(day,month,year)
    timeT="%02d:%02d:%02d"%(hour,minute,second)
    return dateT, timeT,day,month,year,hour,minute,second 
########################################################
emotionTotal=[]
genderTotal=[]
ageTotal=[]
timeTotal=[]
dateTotal=[]
nomberTotal2=[]
timeTotal2=[]
dateTotal2=[]




for filename in os.listdir(Input_directory):    #Main loop on all images in the input folder
    
    
        fpath = os.path.join(Input_directory, filename)
        dateFile,timeFile,day,month,year,hour,minute,second = modification_date_time(fpath)
        try:
            a,resultImg=Detect_face(fpath)

            flag,resultImg,genderFile,ageFile=Age_Gender(fpath,a,resultImg) 
            
            flag2,result,emotionFile= Emotion(fpath,resultImg,a)
            
            FileExtention=Output_directory+"Img_"+str(day)+"_"+str(month)+"_"+str(year)
            FileExtention=FileExtention+"_"+str(hour)+"_"+str(minute)+"_"+str(second)+".jpg"                       
            
            t=[]
            [t.append(timeFile)  for ii in range(len(emotionFile))]
            d=[]
            [d.append(dateFile)  for ii in range(len(emotionFile))]
            dateTotal.extend(d)
            timeTotal.extend(t)
            emotionTotal.extend(emotionFile)
            genderTotal.extend(genderFile)
            ageTotal.extend(ageFile) 
            
            nomberTotal2.append(len(emotionFile))
            timeTotal2.append(timeFile)
            dateTotal2.append(dateFile)
            
            cv2.imwrite(FileExtention,result)
  
        except:
            cv2.imwrite(FileProb_directory,cv2.imread(fpath))

        
    
    




    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
 
            
                
                    
  
                



