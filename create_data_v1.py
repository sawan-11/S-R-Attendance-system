# creating employee data
import cv2, sys, os, shutil
import base64
import tkinter as tk
import numpy as np

# python files importing
from credentials import FaceRecognitionCredentials
import json
with open('config.json', 'r') as c:
    params = json.load(c)["params"]

# path for the dataset folder and model
protoPath = os.path.sep.join(['face_detection_model', "deploy.prototxt"])
modelPath = os.path.sep.join(['face_detection_model', "res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
haar_file = 'haarcascade_frontalface_default.xml'

#%%
# function for checking camera is avaliable or not
def available_webcam():
    webcam=None
    try:
        webcam = cv2.VideoCapture(1)
        (_, im) = webcam.read()
        cv2.imshow('testing',im)
        print('Using External Camera now')
        return 1
    except:
        print('Using Internal Camera now')
        return 0
    finally:
        webcam.release()
        cv2.destroyAllWindows()


# function for the GUI
def popUp(popup_text,pass_text=None):
    root = tk.Tk()
    root.attributes("-topmost", True)
    root.title(params['title'])
    root.geometry('500x100+500+300')
    tk.Label(root, text="%s%s"%(popup_text, pass_text), bg="white", fg="black", font=(None, 15), height=50, width=50).pack()
    root.resizable(0, 0)
    root.after(3000, lambda: root.destroy())
    return root.mainloop()


# function for recording image
def recordFacialImage(name,empId):
    sub_data = name + "," + empId
    path = os.path.join(FaceRecognitionCredentials.DATASET_FOLDER_NAME,sub_data)
    try:
            if  os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=False, onerror=None)
                print("Deleted " + format(path))
    except:
            e = sys.exc_info()[0]
            print("Error: %s" % e)    
            
    os.mkdir(path)
    print("Created " + format(path))  
    (width, height) = (130, 100)    # defining the size of image

    face_cascade = cv2.CascadeClassifier(haar_file)
    # webcam = cv2.VideoCapture(available_webcam()) #'0' is use for my webcam, if you've any other camera attached use '1' like this
    # The program loops until it has 30 images of the face.
    webcam = cv2.VideoCapture(0)
    count = 1
    i=0 #
    while count <= 1:
        (_, im) = webcam.read()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        (h, w) = im.shape[:2]
        
        # construct a blob from the image
        imageBlob = cv2.dnn.blobFromImage(cv2.resize(im, (300, 300)), 1.0, (300, 300),(104.0, 177.0, 123.0), swapRB=False, crop=False)
        # apply OpenCV's deep learning-based face detector to localize faces in the input image
        detector.setInput(imageBlob)
        detections = detector.forward()
        
        # compute the (x, y)-coordinates of the bounding box for the face
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        if type(faces)!=tuple:
            check = faces.shape
            cv2.imshow('Data Insight - Face Recognition', im)
            if check[0] > 1:
                popUp("Multiple Faces detected please re-capture","")
                i+=1
                break
        for (x,y,w,h) in faces:
            cv2.rectangle(im, (startX, startY), (endX, endY),(0, 0, 255), 1)
            face = gray[startY:endY, startX:endX]
            face_resize = cv2.resize(face, (width, height))
            cv2.imwrite('%s/%s.png' % (path,count), face_resize)
            count += 1
        cv2.imshow('Data Insight - Face Recognition', im)
        key = cv2.waitKey(10)
        if key == 27:
            break
    webcam.release()
    cv2.destroyAllWindows()
    dict1 = {}
    encoded_final=[]
    if i==1:
        shutil.rmtree(path, ignore_errors=False, onerror=None)
        print("Multiple Faces detected.")
    else:
        popUp("Face captured for employee: ",name)
        imagesList = os.listdir(path)
        for image in imagesList:
            encoded=base64.b64encode(open(path+'/'+image, "rb").read())
            encoded_final.append(encoded.decode('utf-8'))
        
    dict1["images"]=encoded_final
    dict1["uniqueID"]=str(empId)
    dict1["path"]=str(sub_data)
    return dict1

# %%

