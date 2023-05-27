import cv2
import numpy as np
import face_recognition as fr
import os
from datetime import datetime
import pyttsx3

path = 'images'
image = []
student_name = []
mylist = os.listdir(path)

#storing names of students
for i in mylist:
    current_image = cv2.imread(f"{path}\{i}")
    image.append(current_image)
    student_name.append(os.path.splitext(i)[0])
    
# finding encoding of samples
def findencoding(image):
    encoding_list = []
    for img in image:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encoding = fr.face_encodings(img)[0]
        encoding_list.append(encoding)
    return encoding_list
  
# finding encoding of stored samples
encode_list = findencoding(image)

