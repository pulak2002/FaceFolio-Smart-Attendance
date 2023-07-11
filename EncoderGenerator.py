import cv2
import face_recognition
import pickle
import os
from numba import jit, cuda
from timeit import default_timer as timer
import warnings

warnings.filterwarnings("ignore")


start=timer()
# Importing student images
folderPath = 'D:\\pythonProject\\pythonProject\\Resources\\Training'

pathList = os.listdir(folderPath)
imgList = []
studentIds = []

for path in pathList:
    for sub_path in os.listdir(os.path.join(folderPath, path)):
        add = os.path.join(os.path.join(folderPath, path), sub_path)
        imgList.append(cv2.imread(add))
        studentIds.append(os.path.splitext(path)[0])
    # print(os.path.splitext(path)[0])
print(studentIds)

@jit(target_backend='cuda')		
def findEncodings(imagesList):
    encodeList = []
    for img in imagesList:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)

    return encodeList


print("Encoding Started ...")
encodeListKnown = findEncodings(imgList)
encodeListKnownWithIds = [encodeListKnown, studentIds]
print("Encoding Complete")

file = open("D:\\pythonProject\\pythonProject\\Resources\\EncodeFile.p", 'wb')
pickle.dump(encodeListKnownWithIds, file)
file.close()
print("File Saved")
print("Time: ",timer()-start)