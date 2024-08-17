import streamlit as st
import recognition_helper_pg as helper_pg
import recognition_helper as helper
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer
import av
import time
from datetime import datetime
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from email.utils import formataddr

# st.set_page_config(page_title='Registration Form', layout='wide')
st.subheader("Registration Form")

registration_form_pg = helper_pg.RegistrationForm()
registration_form = helper.RegistrationForm()

# Step 1: Collect person name and role
person_name = st.text_input(label='Name', placeholder='Please Enter First and Last name')
email = st.text_input(label='Email', placeholder='Please Enter EmailID')
passw = st.text_input(label='Password', placeholder='Please Enter Password', type='password')
role = st.selectbox(label='Select your role', options=('Student', 'Teacher'))

wait_time = 30
start_time = time.time()

realTimePrediction = helper.RealTimePrediction()

# callback function
def video_frame_callback(frame):
    global start_time
    img = frame.to_ndarray(format="bgr24")  # 3d numpy array
    prediction_img, embeddings = registration_form.get_embeddings(img)

    # save data to local computer
    if embeddings is not None:
        with open('face_embedding.txt', mode='ab') as f:
            np.savetxt(f, embeddings)

    return av.VideoFrame.from_ndarray(prediction_img, format="bgr24")


webrtc_streamer(key="real_time_prediction", video_frame_callback=video_frame_callback)

# step 3: save data in redis database


def sent_email():
    # Your email and app password
    sender_email = "shreyubanawala@gmail.com"
    sender_name = "Shreyash Banawala"
    sender_password = "wqhd sbgs qokq bfdb"

    # Create message
    msg = MIMEMultipart()
    msg["From"] = formataddr((sender_name, sender_email))
    msg['To'] = email
    msg['Subject'] = "Account Created"
    body = (f"Hi {person_name},<br/><br/>Hope you are doing well, Your account has been created at "
            f"{datetime.now().strftime("%d/%m/%Y %H:%M:%S")}."
            f"<br/><br/><p style='color:red'>Important Note: This is an automatic email. please do not reply this email.</p>"
            f"<br/><br/>Thank you")
    msg.attach(MIMEText(body, 'html'))

    # Send email
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)
        text = msg.as_string()
        server.sendmail(sender_email, email, text)
        server.quit()
        print("Email sent successfully")
    except Exception as e:
        print(f"Error: {str(e)}")


if st.button('Submit'):
    data, response_message = registration_form_pg.save_data_pg(person_name, email, passw, role)
    registration_form.redis_save_data(person_name, email, passw, role)
    if response_message == 'name_false':
        st.error("Please enter the name. Name cannot be empty")
    if response_message == 'email_false':
        st.error("Please enter the email. EmailID cannot be empty")
    elif response_message == 'password_false':
        st.error("Please enter the password. Password cannot be empty")
    elif response_message == 'file_false':
        st.error('Issue in saving face embeddings. Reload the page and try again.')
    elif data == 0:
        st.error(response_message)
    elif response_message == 0:
        st.error("Email ID already exists")
    else:
        st.success(f'{person_name} registered successfully.')
        sent_email()