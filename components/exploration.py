import uuid
from util import add_index_column_if_first_is_float, correlation_matrix_to_table, data_uploader_components, descriptive_stats_to_table, dtype_to_sql, generate_random_alphanumeric, get_data, transform_digits
from streamlit_pandas_profiling import st_profile_report
from util import config, remove_punctuation
import pandas as pd
from datetime import datetime
from sqlalchemy import create_engine
import streamlit as st
# pakai sql alchemy (menggunakan konfigurasi pandas)

def insert_data(dataset_name, dataframe):
    engine = create_engine(f"starrocks://{config['db_user']}:{config['db_password']}@{config['db_host']}:{config['db_port']}/{config['db_name']}")
    connection = engine.connect()
    uid = generate_random_alphanumeric()
    uid = str(uid).replace("-","")

    # Convert DataFrames to JSON strings as necessary
    dataframe_json = dataframe

    # Generate data correlation matrix and convert to JSON
    data_correlation = correlation_matrix_to_table(dataframe)
    
    # Generate data description (summary statistics)
    data_description = descriptive_stats_to_table(dataframe)

    # Establish a connection to the database

    data_insert = {
        'ori': dataframe,
        'descr':data_description,
        'corr':data_correlation
    }

    table_name = {key:f"de_{uid}_{key}_{dataset_name.replace('.csv','')}" for key in data_insert.keys()}

    for key in table_name.keys():
        df = data_insert[key]
        if isinstance(df, pd.Series):
            clean_name = transform_digits(remove_punctuation(df.name).replace(' ','_'))
            sql_type = dtype_to_sql(df.dtype.name)
            col_with_type = f"{clean_name} {sql_type}"
            print(col_with_type)
            create_table_query = f"CREATE TABLE {table_name[key]} ({col_with_type})"
        else:
            df.columns = [transform_digits(remove_punctuation(str(col)).replace(' ', '_')) for col in df.columns]
            valid_columns = [col for col in df.columns if not col.startswith('Unnamed')]
            df = df[valid_columns]
            df = add_index_column_if_first_is_float(df)
            data_insert[key] = df
            cols_with_types = ", ".join([f"{transform_digits(remove_punctuation(col).replace(' ','_'))} {dtype_to_sql(df[col].dtype.name)}" for col in df.columns if not col.startswith('Unnamed')])
            create_table_query = f"CREATE TABLE {table_name[key]} ({cols_with_types})"
        connection.execute(create_table_query)
    
    # Count variables (columns) and observations (rows)
    variable_number = len(dataframe.columns)
    observation_number = len(dataframe)
    
    # Establish a connection to the database

    # Prepare and execute the insertion query
    connection.execute(
        """
        INSERT INTO data_exploration_master (
            id, dataset, variable_number, 
            observation_number, created_at
        ) VALUES (%s, %s, %s, %s, %s)
        """,
        (uid, dataset_name,
         variable_number, observation_number, datetime.now())
    )

    for key in table_name.keys():
        df = data_insert[key]
        if isinstance(df, pd.Series):
            column_name = transform_digits(remove_punctuation(df.name).replace(' ','_'))
            insert_query = f"INSERT INTO {table_name[key]} ({column_name}) VALUES (%s)"
            values_to_insert = [(value,) for value in df]
        else:
            df.columns = [transform_digits(remove_punctuation(str(col)).replace(' ', '_')) for col in df.columns]
            valid_columns = [col for col in df.columns if not col.startswith('Unnamed')]
            column_names = ', '.join([transform_digits(remove_punctuation(col).replace(' ','_')) for col in df.columns if not col.startswith('Unnamed')])
            placeholders = ', '.join(['%s'] * len(valid_columns))
            insert_query = f"INSERT INTO {table_name[key]} ({column_names}) VALUES ({placeholders})"
            values_to_insert = df[[col for col in df.columns if not col.startswith('Unnamed')]].values.tolist()
        connection.execute(insert_query, values_to_insert)
    print("Data input successfully")
    st.success(f"Data saved into database with table name:{','.join([table_name[key] for key in table_name.keys()])}")

def data_exploration_page(st):
    st.markdown("<h2 class='menu-title'>Data Exploration</h2>",
                unsafe_allow_html=True)
    st.markdown("<h6 class='menu-subtitle'>Analyzing and visualizing a dataset to gain a deeper understanding of its characteristics, structure, and potential patterns</h6>",
                unsafe_allow_html=True)
    st.markdown("<hr class='menu-divider' />",
                unsafe_allow_html=True)

    engine = create_engine(f"starrocks://{config['db_user']}:{config['db_password']}@{config['db_host']}:{config['db_port']}/{config['db_name']}")
    connection = engine.connect()

    # Create table if it doesn't exist
    create_table_query = """
    CREATE TABLE IF NOT EXISTS data_exploration_master (
        id VARCHAR(100),
        dataset VARCHAR(50),
        variable_number INT,
        observation_number INT,
        created_at DATETIME
    );
    """
    connection.execute(create_table_query)
    print("Database and table checked/created successfully.")

    data_uploader_components(st)

    # Showing the uploaded file from session state
    try:
        st.write(st.session_state.uploaded_file)
        st.success("The data have been successfully uploaded")

        # Initiating pandas profiling
        if st.button('Plot the Data Exploration'):
            pr = st.session_state.uploaded_file.profile_report()
            st_profile_report(pr)

        if st.button("Save the Data"):
            insert_data(st.session_state['data_name'], st.session_state["uploaded_file"])
            
        else:
            st.write("")
    except Exception as e:
        print(e)
        st.markdown("<span class='info-box'>Please upload any data</span>",
                    unsafe_allow_html=True)

    st.write("")