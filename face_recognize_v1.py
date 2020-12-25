# facerec.py
# importing important modules for face_recognition
import tkinter as tk
import cv2, threading
from keras.models import Model, Sequential
from keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dropout, Activation
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
from imutils.video import FPS
import numpy as np
from os import listdir
from credentials import FaceRecognitionCredentials
import pyodbc
import json
with open('config.json', 'r') as c:
    params = json.load(c)["params"]
#%%

# mssql server and database name
SERVER = "LAPTOP-LVK4MC26\SAWAN"
DATABASE = "attendance"

#%%.
employees = dict()

def available_webcam():
    try:
        webcam = cv2.VideoCapture(1)
        (_, im) = webcam.read()
        cv2.imshow('testing',im)
        webcam.release()
        cv2.destroyAllWindows()
        return 1
    except:
        return 0

color = (0, 0, 255)

#you can refer this https://towardsdatascience.com/face-detection-in-2-minutes-using-opencv-python-90f89d7c0f81
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# here prcocessing the image
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    # preprocess_input normalizes input in scale of [-1, +1]. You must apply same normalization in prediction.
    # Ref: https://github.com/keras-team/keras-applications/blob/master/keras_applications/imagenet_utils.py (Line 45)
    img = preprocess_input(img)
    return img

# vgg model training
def loadVggFaceModel():
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(224, 224, 3)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Convolution2D(4096, (7, 7), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(4096, (1, 1), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(2622, (1, 1)))
    model.add(Flatten())
    model.add(Activation('softmax'))

    # you can download pretrained weights from https://sefiks.com/2018/08/06/deep-face-recognition-with-keras/
    model.load_weights('vgg_face_weights.h5')
    vgg_face_descriptor = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)
    return vgg_face_descriptor

# making variable model
model = loadVggFaceModel()

def model_training():
    employee_list = listdir(FaceRecognitionCredentials.DATASET_FOLDER_NAME)
    global employees
    for file in employee_list:
        #employee, extension = file.split(",")
        employees[file] = model.predict(preprocess_image(FaceRecognitionCredentials.DATASET_FOLDER_NAME+'/'+file+'/1.png'))[0, :]
    print("employee representations data fetched successfully")

model_training() # Calling model_training

def findCosineSimilarity(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))


def calculatePopupCoordinatesAndShow(count,pass_text):
    # This function is created to call pop-up using tkinter.
    mod = count % 4
    root = tk.Tk()
    root.attributes("-topmost", True)
    root.title(params['title'])
    if mod == 1:
        root.geometry('500x100+400+600')
    elif mod == 2:
        root.geometry('500x100+100+300')
    elif mod == 3:
        root.geometry('500x100+400+100')
    else:
        root.geometry('500x100+700+300')
    tk.Label(root, text="Employee identified as: "+pass_text, bg="white", fg="black", font=(None, 15), height=50, width=50).pack()
    root.resizable(0, 0)
    root.after(3000, lambda: root.destroy())
    return root.mainloop()


# this function for identify the employee images and using the model here
def identifyFacialImage():
    count = 0
    # employees = model_training()
    cap = cv2.VideoCapture(0)
    print('Webcam opened')
    prev_val = set()
    def clearList():
        prev_val.clear()

    def set_interval(func, sec):
        def func_wrapper():
            set_interval(func, sec)
            func()
        t = threading.Timer(sec, func_wrapper)
        t.start()

    set_interval(clearList, 60) # 60 sec will hold the identification
    set_interval(model_training, 3600)
    fps = FPS().start()
    while True:
        ret, img = cap.read()
        # img = cv2.resize(img, (640, 360))
        faces = face_cascade.detectMultiScale(img, 1.3, 5)
        for (x, y, w, h) in faces:
                if w > 130:
                    detected_face = img[int(y):int(y + h), int(x):int(x + w)]  # crop detected face
                    detected_face = cv2.resize(detected_face, (224, 224))
                    img_pixels = image.img_to_array(detected_face)
                    img_pixels = np.expand_dims(img_pixels, axis=0)
                    # employee dictionary is using preprocess_image and it normalizes in scale of [-1, +1]
                    img_pixels /= 127.5
                    img_pixels -= 1
                    captured_representation = model.predict(img_pixels)[0, :]
                    for i in employees:
                        new_val = i
                        representation = employees[i]
                        similarity = findCosineSimilarity(representation, captured_representation)
                        #print("similarity",similarity)
                        if (similarity < 0.13):
                            if new_val not in prev_val:
                                print("code to call the webservice")
                                new_list = new_val.split (",")
                                empid = new_list[1]
                                query = "IF NOT EXISTS (Select Id From Attendance_record Where Id='" + new_list[1] + "' and ""cast(InTime as Date) = cast(getdate() as date)) Begin Insert into Attendance_record " "(name,Id,InTime,OutTime,InputPerDay,IsProccessed) Values ""('"+new_list[0]+"','" +new_list[1] + "',GETDATE(),NULL,1,'Y') End ELSE Begin Update Attendance_record" " Set OutTime=getdate(),InputPerDay=InputPerDay+1 Where Id='" +new_list[1] + "' " "and cast(InTime as Date) = cast(getdate() as date) End"
                                print(empid)
                                # print(prev_val)
                                mssqlConnection = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};SERVER='+SERVER+';DATABASE='+DATABASE+'; Trusted_connection=yes;')
                                print('Got the mssqlConnection:',mssqlConnection)
                                with mssqlConnection:
                                    mssqlCursor = mssqlConnection.cursor()
                                    print('Executing the query:',query)
                                    mssqlCursor.execute(query)
                                    mssqlCursor.commit()
                                thread1 = threading.Thread(target = calculatePopupCoordinatesAndShow , args = (count,new_list[0],))
                                thread1.start()
                                count += 1
                                prev_val.add(new_val)
                                print("employee_name",new_val)
                                cv2.putText(img, new_val, (int(x + w + 15), int(y - 12)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                                break
                        else:
                            cv2.putText(img, 'unknown', (int(x + w + 15), int(y - 12)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    # connect face and text
                    cv2.line(img, (int((x + x + w) / 2), y + 15), (x + w, y - 20), color, 1)
                    cv2.line(img, (x + w, y - 20), (x + w + 10, y - 20), color, 1)
        fps.update()
        cv2.imshow('S.R Attendance', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # press q to quit
            # kill open cv things
            fps.stop()
            cap.release()
            cv2.destroyAllWindows()
            break


# Four main thing that we want to record in database
# User_id int
# emp_name varchar
# in_time Datetime
# out_time datetime
# InputPerDay int (whenever the employee detect this will increament by 1)
# IsProccessed char