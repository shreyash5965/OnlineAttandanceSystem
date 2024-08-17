import pandas as pd
import streamlit as st
import recognition_helper as helper

# st.set_page_config(page_title='Reporting', layout='wide')

if 'user_data' in st.session_state:
    st.subheader('Report')
    realTimePrediction = helper.RealTimePrediction()
    key_name = 'attendance:logs'
    # retrieve log data and show in report
    def load_logs(key_name, end=-1):
        logs_list = helper.r.lrange(key_name, start=0, end=end)  # extract all data from redis database
        return logs_list


    # tab to show information
    tab1, tab2, tab3 = st.tabs(['Registered Data', 'Logs', 'Attendance Report'])
    with tab1:
        if st.button('Refresh Data'):
            with st.spinner('Retriving data from Redis database'):
                registration_data = realTimePrediction.retrieve_data()
                st.dataframe(registration_data[['Name', 'EmailID', 'Role', 'Password']])
    with tab2:
        if st.button('Refresh Logs'):
            # retrieve data from redis database
            st.write(load_logs(key_name))

    with tab3:
        st.subheader("Attendance report")
        # load logs in attribute list
        log_list = load_logs(key_name)

        # step 1: convert logs from list of bytes to list of string
        convert_byte_to_string = lambda x: x.decode('utf-8')
        logs_list_string = list(map(convert_byte_to_string, log_list))

        # step 2 : split string by @ and create nested list
        split_string = lambda x: x.split('#')
        logs_nested_list = list(map(split_string, logs_list_string))

        # step 3 : convert nested list into dataframe
        logs_df = pd.DataFrame(logs_nested_list, columns=['Name', 'Role', 'Timestamp'])

        # step : Time based analysis
        # convert the time to timestamp value for further analysis
        logs_df['Timestamp'] = pd.to_datetime(logs_df['Timestamp'])
        logs_df['Date'] = logs_df['Timestamp'].dt.date

        # calculate in and out time
        # In-time : Min timestamp for that person for the given day
        # Out-time : Max timestamp for that person for the given day
        report_df = logs_df.groupby(by=['Date', 'Name', 'Role']).agg(
            In_time=pd.NamedAgg('Timestamp', 'min'),
            Out_time=pd.NamedAgg('Timestamp', 'max')
        ).reset_index()

        report_df['In_time'] = pd.to_datetime(report_df['In_time'])
        report_df['Out_time'] = pd.to_datetime(report_df['Out_time'])

        report_df['Duration'] = report_df['Out_time'] - report_df['In_time']

        # step 5 : Present and Absent marking
        unique_dates = report_df['Date'].unique()
        name_role = report_df[['Name', 'Role']].drop_duplicates().values.tolist()

        date_name_role_zip = []
        for date in unique_dates:
            for name, role in name_role:
                date_name_role_zip.append([date, name, role])

        date_name_role_zip_df = pd.DataFrame(date_name_role_zip, columns=['Date', 'Name', 'Role'])
        # left join the dataframe with report df to find all student/teacher not present in the given day

        date_name_role_zip_df = pd.merge(date_name_role_zip_df, report_df, how='left', on=['Date', 'Name', 'Role'])

        date_name_role_zip_df['Duration Seconds'] = date_name_role_zip_df['Duration'].dt.seconds
        date_name_role_zip_df['Duration Hours'] = date_name_role_zip_df['Duration Seconds'] / 3600

        date_name_role_zip_df['Status'] = date_name_role_zip_df['Duration Hours'].apply(helper.attendance_status)

        st.write(date_name_role_zip_df)

else:
    st.warning('Please login first')