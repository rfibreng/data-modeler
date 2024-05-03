import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from streamlit_option_menu import option_menu
from util import load_result, next_page_inference, prev_page, next_page, prev_page_inference
import os
import pickle
from datetime import datetime, timedelta
from util import config
# import psycopg2
import pandas as pd

def get_total_records(dbname, user, password, host):
    """Fetch the total number of records from the specified table."""
    # Connect to the database
    table_name = 'predictions'
    conn = psycopg2.connect(dbname=dbname, user=user, password=password, host=host)
    cursor = conn.cursor()

    # SQL query to count rows in the table
    query = f"SELECT COUNT(*) FROM {table_name};"

    try:
        cursor.execute(query)
        total_records = cursor.fetchone()[0]  # Fetch the first (and only) row, which is the count
        print(f"Total records in table '{table_name}': {total_records}")
        return total_records
    except psycopg2.Error as e:
        print(f"An error occurred: {e}")
    finally:
        # Close the cursor and the connection
        cursor.close()
        conn.close()

def fetch_all_data(dbname, user, password, host, start_date, end_date, model_name, page, page_limit=25):
    # Connect to the database
    conn = psycopg2.connect(dbname=dbname, user=user, password=password, host=host)
    cursor = conn.cursor()

    # Format the date inputs to strings in the format that SQL expects (YYYY-MM-DD)
    formatted_start_date = start_date.strftime('%Y-%m-%d')
    # Add one day to end_date and subtract one microsecond to include the entire end_date
    extended_end_date = end_date + timedelta(days=1) - timedelta(microseconds=1)
    formatted_end_date = extended_end_date.strftime('%Y-%m-%d %H:%M:%S.%f')

    # Prepare the query with filters and pagination
    query = """
    SELECT id, prediction_at, dataset, model_prediction, output_prediction FROM predictions
    WHERE prediction_at >= %s AND prediction_at <= %s
    AND model_prediction LIKE %s
    ORDER BY prediction_at DESC
    OFFSET %s LIMIT %s;
    """

    # Adjust model name for SQL LIKE query
    model_name_like = '%' + model_name + '%'
    offset = page * page_limit

    try:
        # Execute the query with parameters
        cursor.execute(query, (formatted_start_date, formatted_end_date, model_name_like, offset, page_limit))
        
        # Fetch the rows
        rows = cursor.fetchall()
        
        # Display each row in Streamlit columns
        if rows:
            for index, (id, prediction_at, dataset, model_prediction, output_prediction) in enumerate(rows, start=1 + page * page_limit):
                col1, col2, col3, col4 = st.columns([1, 3, 3, 3])
                col1.write(str(index))
                col2.write(str(prediction_at))
                col3.write(dataset)
                col4.write(model_prediction)

                # Expander to show the output_prediction JSON data
                with st.expander("Show Data"):
                    if output_prediction:
                        # Convert JSON string from database to pandas DataFrame
                        prediction_data = pd.DataFrame(output_prediction)
                        prediction_data = prediction_data.drop(columns=['Unnamed: 0'])
                        st.dataframe(prediction_data)
                    else:
                        st.write("No prediction data available.")
        else:
            st.write("No data available for the selected filters.")
            
    except psycopg2.Error as e:
        st.error(f"An error occurred: {e}")
    finally:
        # Close the cursor and the connection
        cursor.close()
        conn.close()

def inference_log_page(st):
    st.markdown("<h2 class='menu-title'>Inference Logs</h2>",
                unsafe_allow_html=True)
    st.markdown("<h6 class='menu-subtitle'>Access your Inference logs here</h6>",
                unsafe_allow_html=True)
    st.markdown("<hr class='menu-divider' />",
                unsafe_allow_html=True)

    # Adding one space
    st.markdown("<br>", unsafe_allow_html=True)
    # Input for start date, end date, and model name filtering
    col1,col2,col3 = st.columns(3)

    hundred_years_ago = datetime.now() - timedelta(days=365 * 100)
    start_date = col1.date_input("Start date", value=hundred_years_ago)
    end_date = col2.date_input("End date")
    model_name = col3.text_input("Filter by model name")

    st.markdown("<br>", unsafe_allow_html=True)
    print(config["output_path"])
    page_limit = 25
    if 'current_page_inference' not in st.session_state:
        st.session_state.current_page_inference = 0

    total_data = get_total_records('postgres', config['db_user'], config['db_password'], config['db_host'])
    start_index = st.session_state.current_page_inference * page_limit
    end_index = start_index + page_limit
    if total_data is not None:
        if st.session_state.current_page_inference > 0:
                col1.button('Previous', on_click=lambda: prev_page_inference(st))
        if end_index < int(total_data):
            col3.button('Next', on_click=lambda: next_page_inference(st))
        else:
            st.markdown('<br>', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns([1, 3, 3, 3])
        col1.markdown("#### No.")
        col2.markdown("#### Datetime")
        col3.markdown("#### Dataset Name")
        col4.markdown("#### Model Name")
        print(st.session_state.current_page_inference)
                
        fetch_all_data('postgres', config['db_user'], config['db_password'], config['db_host'], start_date, end_date, model_name, st.session_state.current_page_inference, page_limit)
    else:
        col1, col2, col3, col4 = st.columns([1, 3, 3, 3])
        col1.markdown("#### No.")
        col2.markdown("#### Datetime")
        col3.markdown("#### Dataset Name")
        col4.markdown("#### Model Name")