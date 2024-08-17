import streamlit as st
import recognition_helper_pg as helper
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
st.subheader("Login")

registration_form = helper.RegistrationForm()

# Step 1: Collect person name and role
email = st.text_input(label='Email', placeholder='Please Enter EmailID')
passw = st.text_input(label='Password', placeholder='Please Enter Password', type='password')

start_time = time.time()
realTimePrediction = helper.RealTimePrediction()

if 'user_data' not in st.session_state:
    st.session_state.user_data = []

# step 3: save data in redis database


def sent_email():
        # Your email and app password
        sender_email = "shreyubanawala@gmail.com"
        sender_name = "Shreyash Banawala"
        sender_password = "wqhd sbgs qokq bfdb"

        # Create message
        msg = MIMEMultipart()
        msg["From"] = formataddr((sender_name, sender_email))
        msg['To'] = st.session_state.user_data[0]["email"]
        msg['Subject'] = "Login Alert"
        body = (f"Hi {st.session_state.user_data[0]["name"]},<br/><br/>Hope you are doing well, Your account was just logged in from new device at "
                f"{datetime.now().strftime("%d/%m/%Y %H:%M:%S")}."
                f"<br/><br/><p style='color:red'>If you don't recognise this activity. Please contact admin to change your account password."
                f"<br/><br/>Important Note: This is an automatic email. please do not reply this email.</p>"
                f"<br/><br/>Thank you")
        msg.attach(MIMEText(body, 'html'))

        # Send email
        try:
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(sender_email, sender_password)
            text = msg.as_string()
            server.sendmail(sender_email, st.session_state.user_data[0]["email"], text)
            server.quit()
            print("Email sent successfully")
        except Exception as e:
            print(f"Error: {str(e)}")


if st.button('Submit'):
    response = registration_form.check_login(email, passw)
    if not response["isSuccess"]:
        st.error(response["message"])
    else:
        st.success(response["message"])
        st.session_state.logged_in = True
        st.session_state.user_data.append({
            'id': response["data"]["intFaceDataID"],
            'name': response["data"]["strFullName"],
            'role': response["data"]["strRole"],
            'email': response["data"]["strEmailID"]
        })
        sent_email()