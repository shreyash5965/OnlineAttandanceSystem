import streamlit as st

st.set_page_config(page_title='Attendance System', layout='wide')
st.subheader("Online Attendance Attentiveness System using Face Recognition")

with st.spinner("Loading Models and Connecting to Database"):
    import recognition_helper as helper
# st.success("Model Loaded Successfully")

st.write(f"Hello {st.session_state.user_data[0]["name"] if 'user_data' in st.session_state else 'Guest'}")
if 'user_data' in st.session_state:
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = None
        st.experimental_rerun()