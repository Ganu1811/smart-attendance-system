import cv2
import numpy as np
import face_recognition as fr
import os
from datetime import datetime
import pyttsx3

engine = pyttsx3.init()

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

def markattendance(name):
    with open('attendance.csv', 'r+') as f:
        mydatalist = f.readline()
        namelist = []
        for line in mydatalist:
            entry = line.split(',')
            namelist.append(entry[0])

        if name not in namelist:
            now = datetime.now()
            time = now.strftime('%H:%M')
            f.writelines(f'\n{name},{time}')
            

vid = cv2.VideoCapture(0)

# finding encoding of cam video frames
while True:
    success, frames = vid.read()
    #frames = cv2.cvtColor(frames, cv2.COLOR_BGR2RGB)
    faces_in_frames = fr.face_locations(frames)
    encode_in_frames = fr.face_encodings(frames, faces_in_frames)

    for encode_frame, faceloc in zip(encode_in_frames, faces_in_frames):
        match = fr.compare_faces(encode_list, encode_frame)
        facedis = fr.face_distance(encode_list, encode_frame)
        print(facedis)
        matchindex = np.argmin(facedis)        

        # matching stored encoding with video faces
        if match[matchindex]:
            name = student_name[matchindex].upper()
            y1, x2, y2, x1 = faceloc
            y1, x2, y2, x1 = y1+4, x2+4, y2+4, x1+4
            cv2.rectangle(frames, (x1, y1), (x2, y2), (0,255,0), 3)
            cv2.rectangle(frames, (x1, y2-25), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(frames, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            markattendance(name)
            engine.say('welcome to class' + name)
            engine.runAndWait()
            
    cv2.imshow('Video', frames)
    cv2.waitKey(1)

