import os
from datetime import datetime
import cv2
import numpy as np
import pandas as pd
from insightface.app import FaceAnalysis
# similarity and distance calculation
from sklearn.metrics import pairwise
import time
from datetime import datetime
from keras.models import load_model
from openpyxl import Workbook, load_workbook
from scipy.spatial import distance as dist
import streamlit as st
import psycopg2
from psycopg2 import sql

# configer face analysis model
app_sc = FaceAnalysis(name='buffalo_sc', providers=['CPUExecutionProvider'])
app_sc.prepare(ctx_id=0, det_size=(640, 640))

def connect_to_postgres():
    try:
        connection = psycopg2.connect(
        dbname = "Face Recognition",
        user = "postgres",
        password = "postgres",
        host = "localhost",
        port = "5432"
        )

        return connection

    except Exception as error:
        print("Error while connecting to PostgreSQL", error)
        return None


# name searching algorithm
def mL_search_algorithm(df, feature_column, test_data, threshold=0.5):
    # 1. copy the dataframe and prepare for next step
    df_temp = df.copy()

    # 2. get face embeding from dataframe and prepare X(pretrained data) and y(test_data)
    X_list = df[feature_column].tolist()
    X = np.asarray(X_list)

    # reshape data to 1 row and as many columns as necessary
    y = test_data.reshape(1, -1)

    # 3. Calculate cosine similarity
    similarity = pairwise.cosine_similarity(X, y)
    similarity_array = np.array(similarity).flatten()
    df['cosine'] = similarity_array
    # 4. Filter the data with highest Cosine similarity
    person_name, role = '', ''
    filter_query = df['cosine'] > threshold
    filtered_data = df[filter_query]
    # print(f"df Copy: {df['cosine']}")
    if len(filtered_data) > 0:
        filtered_data.reset_index(drop=True, inplace=True)
        # argmax returns index of maxium_value
        cosine_argmax = filtered_data['cosine'].argmax()
        # print(f"cosine_argmax: {cosine_argmax}")
        person_name, role = filtered_data.loc[cosine_argmax][['Name', 'Role']]
    else:
        person_name, role = 'Unknown', 'Unknown'

    return person_name, role


def attendance_status(duration):
    if pd.Series(duration).isnull().all():
        return 'Absent'
    elif duration >= 0 and duration < 1:
        return 'Absent (< 1hr)'
    elif duration >= 1 and duration < 4:
        return 'Half Day (<4 hrs)'
    else:
        return 'Present'


# save data/logs every minute
class RealTimePrediction:
    def __init__(self):
        self.data_dict = dict(name=[], role=[], current_time=[])
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        self.emotion_model = load_model('./../emotion_model.h5')

    def reset_dict(self):
        self.data_dict = dict(name=[], role=[], current_time=[])

    def save_logs(self, id, subject, emotion, isDistracted, attentiveNumber):
        # print(f"Data: {subject} - {emotion} - {isDistracted}")
        connection = connect_to_postgres()

        if connection is not None:
            cursor = connection.cursor()
            try:
                # Call the stored procedure
                sql_command = """
                                SELECT * FROM sp_operations(%s,%s, %s, %s, %s, %s, %s, %s);
                            """
                params = (4, 'ref1', id, subject, emotion, str(attentiveNumber), str(isDistracted), '')
                cursor.execute(sql_command, params)
                cursor.execute("FETCH ALL IN ref1;")
                result = cursor.fetchall()
                result = list(result[0])
                # # Commit the transaction
                cursor.execute("COMMIT;")
                # print(f"result: {result}")
                return result[0]

            except Exception as error:
                return error
            finally:
                cursor.close()
                connection.close()

    # Function to calculate eye aspect ratio (EAR)
    def eye_aspect_ratio(eye):
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear

    def name_prediction(self, image, df):
        # print("Inside name prediction ")

        start_time = time.time()
        last_active_time = time.time()
        distraction_threshold = 5  # in seconds
        distracted = False

        now = datetime.now()
        userDataArray = []
        image_information = app_sc.get(image)
        image_copy = image.copy()
        current_time = str(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(gray_frame, 1.3, 5)
        # print(f"Faces: {faces}")
        person_name = ""
        # emotion = ""
        # for info in faces:
        #     print(f"X,Y,W,H: {info}")
            # face = gray_frame[y:y + h, x:x + w]
            # face = cv2.resize(face, (48, 48))
            # face = face.astype('float32') / 255.0
            # face = np.expand_dims(face, axis=0)
            # face = np.expand_dims(face, axis=-1)
            #
            # emotion_prediction = emotion_model.predict(face)
            # max_index = np.argmax(emotion_prediction[0])
            # emotion = emotion_labels[max_index]

        for info in image_information:
            embeddings = info['embedding']
            person_name, role = mL_search_algorithm(df, 'Facial Features', embeddings)
            x, y, w, h = info['bbox'].astype(int)
            print(f"Info: {info['bbox'].astype(int)}")
            # set text_color green if the person name is present else red
            # text_color = (0, 255, 0) if person_name != 'Unknown' else (0, 0, 255)
            # display rectangle around the face
            # cv2.rectangle(image_copy, (x1, y1), (x2, y2), text_color, 2)
            # # display name of person
            # cv2.putText(image_copy, person_name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 0.5, text_color)
            # # display the current date time
            # cv2.putText(image_copy, current_time, (x1, y2 + 20), cv2.FONT_HERSHEY_DUPLEX, 0.5, text_color)
            #
            # self.data_dict['name'].append(person_name)
            # self.data_dict['role'].append(role)
            # self.data_dict['current_time'].append(current_time)

            face = gray_frame[y:y + h, x:x + w]
            face = cv2.resize(face, (48, 48))
            face = face.astype('float32') / 255.0
            face = np.expand_dims(face, axis=0)
            face = np.expand_dims(face, axis=-1)
            emotion_prediction = self.emotion_model.predict(face)
            max_index = np.argmax(emotion_prediction[0])
            emotion = self.emotion_labels[max_index]

            # person_name = self.name_prediction(image_copy, decoded_df)
            text_color = (0, 255, 0) if person_name != 'Unknown' else (0, 0, 255)

            # Draw rectangle around detected face
            person_data = f"Name: {person_name}\nEmotion: {emotion}"
            person_data = person_data.split('\n')

            for i, person_data in enumerate(person_data):
                cv2.putText(image_copy, person_data, (x, (y - 20) + i * 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color)

            cv2.rectangle(image_copy, (x, y), (w, h), text_color, 2)
            roi_gray = gray_frame[y:y + h, x:x + w]

            # Detect eyes within the face region
            eyes = self.eye_cascade.detectMultiScale(roi_gray)

            # Calculate attentiveness based on eye aspect ratio (EAR)
            attentiveness = 0
            for (ex, ey, ew, eh) in eyes:
                eye = roi_gray[ey:ey + eh, ex:ex + ew]
                A = dist.euclidean(eye[1], eye[5])
                B = dist.euclidean(eye[2], eye[4])
                C = dist.euclidean(eye[0], eye[3])
                ear = (A + B) / (2.0 * C)
                attentiveness += ear

            # Normalize attentiveness by the number of eyes
            if len(eyes) > 0:
                attentiveness /= len(eyes)

            # Check if the user is distracted
            if attentiveness < 0.35:  # Example threshold for closed eyes
                if time.time() - last_active_time > distraction_threshold:
                    distracted = True
            else:
                distracted = False

            if person_name != "Unknown":
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                userDataArray = [person_name, timestamp, distracted, emotion, (attentiveness)]

        if len(image_information) < 1 and person_name != 'Unknown':
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            userDataArray = [person_name, timestamp, True, "", 0]
            # sheet.append([person_name, timestamp, True, "", "[]"])

        # return person_name
        return image_copy, userDataArray

    def retrieve_data(self):

        connection = connect_to_postgres()

        if connection is not None:
            cursor = connection.cursor()
            try:
                # Call the stored procedure
                sql_command = """
                                SELECT * FROM sp_operations(%s,%s, %s, %s, %s, %s, %s, %s);
                            """
                params = (0, 'ref1', 0, '', '', '', '', '')
                cursor.execute(sql_command, params)
                cursor.execute("FETCH ALL IN ref1;")

                columns = [desc[0] for desc in cursor.description]
                rows = cursor.fetchall()
                df = pd.DataFrame(rows, columns=columns)
                df['Facial Features'] = df['Facial Features'].apply(lambda x: np.frombuffer(x, dtype=np.float32))
                # print(f"Data: {df}")
                return df

            except Exception as error:
                return 0, error
            finally:
                cursor.close()
                connection.close()

    def converted_data(self):
        # retrieve data from redis
        decoded_dictionary = r.hgetall(name=db_key)
        decoded_series = pd.Series(decoded_dictionary)
        # Convert each element of retrive_series from bytes to 32-bit float NumPy arrays
        decoded_series = decoded_series.apply(lambda x: np.frombuffer(x, dtype=np.float32))

        # Retrieve the index of retrive_series
        index = decoded_series.index

        # Convert the index elements from bytes to strings
        index = list(map(lambda x: x.decode(), index))
        decoded_series.index = index
        # print(f"Redis: {decoded_series}")
        decoded_df = decoded_series.to_frame().reset_index()
        decoded_df.columns = ['name_role', 'facial_features']
        decoded_df.rename(columns={'facial_features': 'Facial Features'}, inplace=True)
        # Split by '#' and create a new DataFrame with two columns 'Name' and 'Role'
        temp_df = decoded_df['name_role'].str.split('#', expand=True)

        # Split each part by '_' and capitalize each word
        decoded_df[['Name', 'Role', 'EmailID', 'Password']] = temp_df.map(
            lambda x: ' '.join([word.capitalize() for word in x.split('_')]) if x is not None else None)
        decoded_df['Password'] = decoded_df['Password'].apply(self.mask_password)
        # print(decoded_df[['Name', 'Role', 'EmailID', 'Password', 'Facial Features']])
        return decoded_df[['Name', 'Role', 'EmailID', 'Password', 'Facial Features']]

    def get_log_data(self, p_id, p_date, p_subject):
        print(p_id , p_date, p_subject)
        connection = connect_to_postgres()

        if connection is not None:
            cursor = connection.cursor()
            try:
                # Call the stored procedure
                sql_command = """
                                SELECT * FROM sp_operations(%s,%s, %s, %s, %s, %s, %s, %s);
                            """
                params = (3, 'ref1', p_id, p_subject, '', '', p_date, '')
                cursor.execute(sql_command, params)
                cursor.execute("FETCH ALL IN ref1;")

                columns = [desc[0] for desc in cursor.description]
                rows = cursor.fetchall()
                df = pd.DataFrame(rows, columns=columns)
                return df

            except Exception as error:
                return 0, error
            finally:
                cursor.close()
                connection.close()

class RegistrationForm:
    def __init__(self):
        self.sample = 0

    def reset(self):
        self.sample = 0

    def get_embeddings(self, image):
        # step 2: take image and apply insightface model
        embedding_list = []
        image_information = app_sc.get(image)
        image_copy = image.copy()
        for info in image_information:
            self.sample += 1
            text = f'Frame recorded {self.sample}'
            embedding = info['embedding']
            embedding_list.append(embedding)
            x1, y1, x2, y2 = info['bbox'].astype(int)
            # set text_color green if the person name is present else red
            text_color = (0, 255, 255)
            # display rectangle around the face
            cv2.rectangle(image_copy, (x1, y1), (x2, y2), text_color, 2)
            cv2.putText(image_copy, text, (x1, y2 + 20), cv2.FONT_HERSHEY_DUPLEX, 0.5, text_color)
        return image_copy, embedding_list

    def save_data_pg(self, p_fullname, p_emailid, p_password, p_role):

        if p_fullname is None or p_fullname.strip() == '':
            return 0, 'name_false'
        elif p_emailid is None or p_emailid.strip() == '':
            return 0, 'email_false'
        elif p_password is None or p_password.strip() == '':
            return 0, 'password_false'
        elif 'face_embedding.txt' not in os.listdir():
            return 0, 'file_false'

        # step 1: Load face_embedding.txt
        x_array = np.loadtxt('face_embedding.txt', dtype=np.float32)  # flattened array
        # step 2: convert to array (proper shape)
        samples = int(x_array.size / 512)
        x_array = x_array.reshape(samples, 512)
        x_array = np.asarray(x_array)

        # step 3: calculate mean embeddings
        x_mean = x_array.mean(axis=0)
        x_mean = x_mean.astype(np.float32)
        p_facedata = x_mean.tobytes()

        # step 4 : Save data to Postgreql

        connection = connect_to_postgres()

        if connection is not None:
            cursor = connection.cursor()
            try:
                # Call the stored procedure
                sql_command = """
                                SELECT * FROM sp_operations(%s,%s, %s, %s, %s, %s, %s, %s);
                            """
                params = (2, 'ref1', 0 ,p_fullname, p_emailid, p_password, p_role, p_facedata)
                cursor.execute(sql_command, params)
                cursor.execute("FETCH ALL IN ref1;")
                result = cursor.fetchall()
                result = list(result[0])
                # # Commit the transaction
                cursor.execute("COMMIT;")
                # print(f"result: {result}")
                return 1, result[0]

            except Exception as error:
                return 0, error
            finally:
                cursor.close()
                connection.close()

    def check_login(self, email, passw):

        connection = connect_to_postgres()

        if connection is not None:
            cursor = connection.cursor()
            try:
                # Call the stored procedure
                sql_command = """
                                SELECT * FROM sp_operations(%s,%s, %s, %s, %s, %s, %s, %s);
                            """
                params = (1, 'ref1', 0, '', email, passw, '', '')
                cursor.execute(sql_command, params)
                cursor.execute("FETCH ALL IN ref1;")
                # result = cursor.fetchall()
                columns = [desc[0] for desc in cursor.description]
                rows = cursor.fetchall()
                data_as_dict = [dict(zip(columns, row)) for row in rows]
                # print(f"result: {data_as_dict}")
                if len(data_as_dict) == 1:
                    return {"isSuccess": True, "message": "Login Successful", "data": data_as_dict[0]}
                elif len(data_as_dict) == 1:
                    return {"isSuccess": False, "message": "One or more account found. Please contact admin", "data": {}}
                else:
                    return {"isSuccess": False, "message": "Incorrect email or password", "data": {}}

            except Exception as error:
                return 0, error
            finally:
                cursor.close()
                connection.close()