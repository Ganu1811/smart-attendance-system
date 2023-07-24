import cv2
import numpy as np
import face_recognition as fr
import os
import pickle
import streamlit as st
from datetime import datetime
import pyttsx3
import pandas as pd
import pywhatkit as pwk


def read_student_phone_numbers(file_path):
    df = pd.read_csv(file_path)
    return df.set_index('Name')['Phone'].to_dict()

# Example usage:
phone_numbers = read_student_phone_numbers('phone_numbers.csv')

# Function to unpickle the file with different encodings
def load_pickle(file_path):
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except UnicodeDecodeError:
        with open(file_path, 'rb') as f:
            return pickle.load(f, encoding='latin1')
    except Exception as e:
        raise e

# Load existing encodings if available, else create empty lists
try:
    encode_list = load_pickle('encodings.pickle')
except FileNotFoundError:
    encode_list = []

path = 'images'
image = []
student_name = []
mylist = os.listdir(path)

for i in mylist:
    current_image = cv2.imread(os.path.join(path, i))
    image.append(current_image)
    student_name.append(os.path.splitext(i)[0])

# finding encoding of stored samples
def findencoding(image):
    encoding_list = []
    for img in image:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encoding = fr.face_encodings(img)[0]
        encoding_list.append(encoding)
    return encoding_list

# Update the encode_list with new encodings
new_encodings = findencoding(image)
encode_list.extend(new_encodings)

# Save encodings using pickle.dump()

with open('encodings.pickle', 'wb') as f:
    pickle.dump(encode_list, f)

def mark_attendance(name):
    with open('attendance.csv', 'a') as f:
        now = datetime.now()
        date_time = now.strftime('%Y-%m-%d %H:%M')
        f.write(f'{name},{date_time}\n')
    engine = pyttsx3.init()
    engine.say(f'Your attendance is marked, Mr. {name}')
    engine.runAndWait()

    if name in phone_numbers:
             phone_number = phone_numbers[name]
             message = f'Mr. {name}. Welcome to class! Your attendance is marked, on {date_time} '
             pwk.sendwhatmsg_instantly(str(phone_number), message)
    else:
             st.write(f'Error: Phone number not found for {name}.')

# Streamlit UI code
st.title('Smart Attendance System')

# Use st.session_state to store the state of the checkbox
if 'start_stop_attendance_system' not in st.session_state:
    st.session_state.start_stop_attendance_system = False

vid = cv2.VideoCapture(0)
present_students = {}

while True:
    start_attendance = st.button('mark your attendance')

    if start_attendance:
        st.session_state.start_stop_attendance_system = True
        st.write('Attendance system is active...')

    if st.session_state.start_stop_attendance_system:
        success, frames = vid.read()
        faces_in_frames = fr.face_locations(frames)
        encode_in_frames = fr.face_encodings(frames, faces_in_frames)

        new_present_students = []

        for encode_frame, faceloc in zip(encode_in_frames, faces_in_frames):
            match = fr.compare_faces(encode_list, encode_frame)
            facedis = fr.face_distance(encode_list, encode_frame)
            print(facedis)

            # Check if any matches were found
            if any(match):
                matchindex = np.argmin(facedis)
                name = student_name[matchindex].upper()

                if name not in present_students:
                    mark_attendance(name)
                    present_students[name] = True
                    st.write(f'Welcome to class, {name}. Your attendance was marked.')

        cv2.imshow('Video', frames)

        # Break out of the loop when attendance is marked
        if present_students:
            break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()
