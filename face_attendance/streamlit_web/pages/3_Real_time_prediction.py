import streamlit as st
import pandas as pd
import recognition_helper_pg as helper_pg
import recognition_helper as helper
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av
import time
from datetime import datetime
from openpyxl import Workbook, load_workbook
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from email.utils import formataddr

if 'user_data' in st.session_state:
    user_data = st.session_state.user_data[0]
    st.session_state.start_time = None
    if "button_clicked" not in st.session_state:
        st.session_state.button_clicked = False
        st.session_state.option_selected = None

    rtc_config = {
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }
    # db_key = helper.db_key
    realTimePrediction = helper.RealTimePrediction()
    realTimePrediction_pg = helper_pg.RealTimePrediction()

    # st.set_page_config(page_title='Prediction', layout='wide')
    st.subheader('Real-time Attendance System')

    with st.spinner("Retrieving Data"):
        # retrieve data from database
        # facial_data_df = realTimePrediction_pg.retrieve_data()
        facial_data_df = realTimePrediction.retrieve_data()
        st.dataframe(facial_data_df)
    # st.success("Data loaded successfully")

    # Real Time Prediction

    # time
    wait_time = 15
    start_time = time.time()
    # count = 0
    user_excel_data = []
    selected_class = ""

    options = ["CPS 2011 - Career Preparation and Success",
               "AML 3406 - AI and ML Capstone Project",
               "AML 3304 - Software Tools and Emerging Technologies for AI and ML",
               "AML 3204 - Social Media Analytics",
               "AML 3104 - Neural Networks and Deep Learning"]
    selected_option = st.selectbox("Select Subject:", options)

    def on_video_started():
        print(f'Started: st.session_state.start_time: {st.session_state.start_time}')
        if st.session_state.start_time is None:
            st.session_state.start_time = datetime.fromtimestamp(time.time())
            print(f"Inside Started {st.session_state.start_time}")
        else:
            st.session_state.start_time = None

    def on_video_stopped():
        print(f'Ended: st.session_state.start_time: {st.session_state.start_time}')
        if st.session_state.start_time is not None:
            print(f"Inside Ended {st.session_state.start_time}")
            sent_email()
            st.session_state.start_time = None
        # print("The video has stopped.")
        # You can also perform other actions here, like releasing resources

    # callback function
    def video_frame_callback(frame):
        global start_time
        global user_excel_data
        # global count
        img = frame.to_ndarray(format="bgr24")  # 3d numpy array
        # prediction_img, data = realTimePrediction.name_prediction(img, facial_data_df)
        prediction_img, data = realTimePrediction_pg.name_prediction(img, facial_data_df)

        time_now = time.time()
        time_difference = time_now - start_time
        # print(f"Start time: {start_time}")
        # print(f"Time Difference: {time_difference}")
        # count = count + 1
        # print(f"Capturing prediction image {count}")

        # print(f"User Data: {data}")
        if time_difference >= wait_time:
            if len(data) > 0 and (data[0] == "" or data[0] == user_data["name"]):
                realTimePrediction_pg.save_logs(user_data["id"], selected_option, data[3], data[2], data[4])
                start_time = time.time()  # reset time
                print("Save data to redis database")
        return av.VideoFrame.from_ndarray(prediction_img, format="bgr24")

    def sent_email():
        started_on = session_state.start_time.strftime('%Y-%m-%d %H:%M:%S')
        df = realTimePrediction_pg.get_log_data(user_data["id"], started_on, st.session_state.option_selected)

        started_on = session_state.start_time.strftime('%Y_%m_%d_%H_%M_%S')
        excel_file = session_state.user_data[0]["name"] + '_' + started_on + '.xlsx'

        sender_email = "shreyubanawala@gmail.com"
        sender_name = "Shreyash Banawala"
        sender_password = "wqhd sbgs qokq bfdb"
        recipient_email = st.session_state.user_data[0]["email"]

        # Create message
        msg = MIMEMultipart()
        msg["From"] = formataddr((sender_name, sender_email))
        msg['To'] = recipient_email
        msg['Subject'] = "Today's attendance report"
        body = f"Hi {st.session_state.user_data[0]["name"]},\n\nHope you are doing well, file below contains the data of your last class.\n\n\nThank you"
        msg.attach(MIMEText(body, 'plain'))

        if len(df) > 0:
            df.to_excel(excel_file, index=False)

            attachment = MIMEBase("application", "octet-stream")
            with open(excel_file, "rb") as file:
                attachment.set_payload(file.read())
            encoders.encode_base64(attachment)
            attachment.add_header(
                "Content-Disposition",
                f"attachment; filename= {excel_file}",
            )
            msg.attach(attachment)

        # Send email
        try:
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(sender_email, sender_password)
            text = msg.as_string()
            server.sendmail(sender_email, recipient_email, text)
            server.quit()
            print("Email sent successfully")
        except Exception as e:
            print(f"Error: {str(e)}")
        finally:
            if os.path.exists(excel_file):
                os.remove(excel_file)

    def on_button_click():
        st.session_state.button_clicked = True
        st.session_state.option_selected = selected_option

    if not st.session_state.button_clicked:
        st.button("Submit", on_click=on_button_click)
    else:
        st.success(f"Currently you are attending class: {st.session_state.option_selected}")

    ctx = webrtc_streamer(
        key="real_time_prediction",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_config,
        video_frame_callback=video_frame_callback,
    )

    if 'prev_state' not in st.session_state:
        st.session_state.prev_state = None
        print(f"st.session_state.prev_state: {st.session_state.prev_state}")

    if ctx.state.playing:
        on_video_started()
    else:
        on_video_stopped()
        # sent_email()

    # if not ctx.state.playing:
    #     if len(st.session_state.user_data[0]["attentiveData"]) > 0:
    #         sent_email()

    # webrtc_streamer(key="real_time_prediction", video_frame_callback=video_frame_callback)

else:
    st.warning('Please login first')
