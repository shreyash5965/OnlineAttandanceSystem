import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode

# Define the RTC configuration (optional, but can be customized)
rtc_config = {
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
}

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    # Perform any processing on the frame (e.g., apply a machine learning model for predictions)
    return frame

def on_video_started():
    print("The video has started.")
    # You can also perform other actions here, like initializing resources

def on_video_stopped():
    print("The video has stopped.")
    # You can also perform other actions here, like releasing resources

st.title("WebRTC Stream Control")

ctx = webrtc_streamer(
    key="real_time_prediction",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=rtc_config,
    video_frame_callback=video_frame_callback,
)

# Use Streamlit's session state to track the previous state
if 'prev_state' not in st.session_state:
    st.session_state.prev_state = None
    print(f"st.session_state.prev_state: {st.session_state.prev_state}")

if ctx.state.playing:
    on_video_started()
else:
    on_video_stopped()

import streamlit as st
import pandas as pd
import numpy as np

# Sample data: distraction level over time
dates = pd.date_range("2024-01-01", periods=100, freq="T")
distraction_level = np.random.randint(0, 100, size=100)

# Create a DataFrame
df = pd.DataFrame({"Time": dates, "Distraction Level": distraction_level})

# Line chart to show distraction level over time
st.line_chart(df.set_index("Time"))