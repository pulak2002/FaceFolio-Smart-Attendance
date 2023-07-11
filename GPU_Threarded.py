import face_recognition
import cv2
import numpy as np
import os
import pickle
import numpy as np
import csv
import face_recognition
import cvzone
import numpy as np
from datetime import datetime
import pandas as pd
import threading
from time import sleep
from threading import Thread
import numba
from numba import jit, cuda
from timeit import default_timer as timer
import warnings

warnings.filterwarnings("ignore")


imgBackground = cv2.imread('D:\\pythonProject\\pythonProject\\Resources\\background.png')
imgBackground_1 = cv2.imread('D:\\pythonProject\\pythonProject\\Resources\\blank_back.png')


folderModePath = 'D:\\pythonProject\\pythonProject\\Resources\\Modes'
modePathList = os.listdir(folderModePath)
imgModeList = []
for path in modePathList:
    imgModeList.append(cv2.imread(os.path.join(folderModePath, path)))



# Load the encoding file
print("Loading Encode File ...")
file = open('D:\\pythonProject\\pythonProject\\Resources\\EncodeFile.p', 'rb')
encodeListKnownWithIds = pickle.load(file)
file.close()
encodeListKnown, studentIds = encodeListKnownWithIds
encodings_array = np.array(encodeListKnown)
# print(studentIds)
print("Encode File Loaded")

global face_names

# Initialize some variables
face_locations = []
face_encodings = []

timestamp = []


now = datetime.now()
current = now.strftime("%d-%m-%Y")

df = pd.DataFrame(list())
df.to_csv('D:\\pythonProject\\pythonProject\\' + current + '.csv')

file = open('D:\\pythonProject\\pythonProject\\' + current + '.csv', 'w+', newline = '')
file_writer = csv.writer(file)

class CustomThread(Thread):
    # constructor

    #@jit(target_backend='cuda')		
    def __init__(self,frame):
        # execute the base constructor
        Thread.__init__(self)
        # set a default value
        self.frame = frame
        self.img_bck_mark =  imgBackground
        self.img_bck_active = imgBackground_1

        self.flag = 1

    #@jit(target_backend='cuda')			
    def run(self):
        frame = self.frame
        face_names = []
        # Only process every other frame of video to save time
        process_this_frame = True
        if process_this_frame:
            # Resize frame of video to 1/4 size for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = small_frame[:, :, ::-1]

            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            if len(face_locations) > 0:
                imgBackground[162:162 + 480, 55:55 + 640] = frame

                predictions = np.zeros(len(studentIds))

                for face_encoding in face_encodings:
                    name = "Unknown"
                    # See if the face is a match for the known face(s)
                    matches = face_recognition.compare_faces(encodeListKnown, face_encoding,tolerance=0.45)
                    face_distances = face_recognition.face_distance(encodeListKnown, face_encoding)

                    # Compute the distance between the face encoding and each image encoding
                    distances = np.linalg.norm(encodings_array - face_encoding, axis=1)

                    # Compute the cosine similarity between the face encoding and each image encoding
                    similarities = np.dot(encodings_array, face_encoding) / (np.linalg.norm(encodings_array, axis=1) * np.linalg.norm(face_encoding))

                    # Combine the distance and similarity measures
                    weights = 1 - distances / np.max(distances)
                    weights = np.clip(weights, 0, 1)
                    similarities = similarities * weights

                    # Compute the indices of the k highest similarities
                    k = 3
                    indices = np.argpartition(similarities, -k)[-k:]

                    # Increment the prediction count for each name at the corresponding indices
                    for index in indices:
                        predictions[index] += similarities[index]

                    best_match_index = np.argmin(face_distances)

                    if matches[best_match_index]:
                        # Determine the predicted name with the highest count
                        max_index = np.argmax(predictions)
                        #name = studentIds[max_index]
                        name = studentIds[best_match_index]

                        imgBackground[44:44 + 633, 808:808 + 414] = cv2.imread('D:\\pythonProject\\pythonProject\\Resources\\Modes\\marked.png')
                        cvzone.putTextRect(imgBackground, name , (875, 600))
                        #cv2.imshow("Face Attendance", imgBackground)
                        self.img_bck_mark = imgBackground
                        self.flag = 0
                        face_names.append(name)
                    elif name == "Unknown":
                        imgBackground_1[44:44 + 633, 808:808 + 414] = cv2.imread('D:\\pythonProject\\pythonProject\\Resources\\Modes\\active.png')
                        #cv2.imshow("Face Attendance", imgBackground)
                        self.img_bck_active = imgBackground_1
                        self.flag = 1
                        cvzone.putTextRect(imgBackground_1, "Unknown", (900, 600))
                        #font = cv2.FONT_HERSHEY_DUPLEX
                        #cv2.putText(imgBackground_1, "Unknown", (950, 600), font, 0.8, (0, 255, 0), 1)
                        continue

                    # Display the results
                    now = datetime.now()
                    date_time = now.strftime("%d/%m/%Y %H:%M:%S")
                    if name == "Unknown":
                        continue
                    file_writer.writerow([name, date_time])

            else:
                imgBackground_1[44:44 + 633, 808:808 + 414] = cv2.imread('D:\\pythonProject\\pythonProject\\Resources\\Modes\\active.png')
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(imgBackground_1, "Stand In front of Webcam", (845,460), font, 0.8, (0, 255, 0), 1)
                cv2.imshow("Face Attendance", imgBackground_1)


        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.8, (255, 255, 255), 1)

        process_this_frame =  process_this_frame
        self.frame = frame


#@jit(target_backend='cuda')			
if __name__ == "__main__":
    # Get a reference to webcam #0 (the default one)
    video_capture = cv2.VideoCapture(0)
    print("Timer start")
    start=timer()
    while True:
        # Grab a single frame of video
        ret, framee = video_capture.read()

        # create a new thread
        thread = CustomThread(framee)
        # start the thread
        thread.start()
        # wait for the thread to finish
        thread.join()
        # get the value returned from the thread
        img = thread.frame

        img_bck_mark = thread.img_bck_mark
        img_bck_active = thread.img_bck_active

        flag = thread.flag

        if flag==0:
            cv2.imshow('Face_Attendance', img_bck_mark)
        else:
            cv2.imshow('Face_Attendance', img_bck_active)

        # Display the resulting image
        cv2.imshow('Video', img)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()
    print("Time: ", timer()-start)

    file.close()

    df = pd.read_csv("D:\\pythonProject\\pythonProject\\" + current + ".csv")
    new_df = pd.DataFrame(list())
    new_df.loc[:,"Name"] =list()
    new_df.loc[:,"Timestamp"] = list()
    res = df.iloc[:,0]
    final_index_occ = []
    for itr in range(len(df)):
        index=max(index for index, item in enumerate(res) if item == df.iloc[itr,0])
        if index in final_index_occ:
            continue
        else:
            final_index_occ .append( index)
    print( final_index_occ)
    for final_index in final_index_occ:
        new_df.loc[len(new_df)+1] = [df.iloc[final_index,0],df.iloc[final_index, 1]]

    new_df.to_csv("D:\\pythonProject\\pythonProject\\" + current + ".csv")
    print(new_df.head())
