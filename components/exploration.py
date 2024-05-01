import uuid
from util import correlation_matrix_to_table, data_uploader_components, descriptive_stats_to_table, dtype_to_sql, get_data, transform_digits
from streamlit_pandas_profiling import st_profile_report
import psycopg2
from util import config, remove_punctuation
import pandas as pd
from datetime import datetime
from psycopg2.extras import execute_values

def insert_data(conn, dataset_name, dataframe):
    uid = uuid.uuid4()
    uid = str(uid).replace("-","")

    cursor = conn.cursor()

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
        else:
            cols_with_types = ", ".join([f"{transform_digits(remove_punctuation(col).replace(' ','_'))} {dtype_to_sql(df[col].dtype.name)}" for col in df.columns if not col.startswith('Unnamed')])
        create_table_query = f"CREATE TABLE {table_name[key]} ({cols_with_types})"
        cursor.execute(create_table_query)
    
    # Count variables (columns) and observations (rows)
    variable_number = len(dataframe.columns)
    observation_number = len(dataframe)
    
    # Establish a connection to the database
    cursor = conn.cursor()

    # Prepare and execute the insertion query
    cursor.execute(
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
            insert_query = f"INSERT INTO {table_name[key]} ({column_name}) VALUES %s"
            execute_values(cursor, insert_query, [(value,) for value in df])
        else:
            insert_query = f"INSERT INTO {table_name[key]} ({', '.join([transform_digits(remove_punctuation(col).replace(' ','_')) for col in df.columns if not col.startswith('Unnamed')])}) VALUES %s"
            execute_values(cursor, insert_query, df[[col for col in df.columns if not col.startswith('Unnamed')]].values.tolist())
    print("Data input successfully")
    
    # Commit the transactions and close the connection
    conn.commit()
    cursor.close()
    conn.close()

def data_exploration_page(st):
    st.markdown("<h2 class='menu-title'>Data Exploration</h2>",
                unsafe_allow_html=True)
    st.markdown("<h6 class='menu-subtitle'>Analyzing and visualizing a dataset to gain a deeper understanding of its characteristics, structure, and potential patterns</h6>",
                unsafe_allow_html=True)
    st.markdown("<hr class='menu-divider' />",
                unsafe_allow_html=True)
    
    conn = psycopg2.connect(dbname='postgres', user=config['db_user'], password=config['db_password'], host=config['db_host'])
    cursor = conn.cursor()

    cursor.execute("CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\";")
    # Create table if it doesn't exist
    create_table_query = """
    CREATE TABLE IF NOT EXISTS data_exploration_master (
        id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
        dataset VARCHAR(50),
        variable_number INT,
        observation_number INT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
    cursor.execute(create_table_query)
    conn.commit()  # Commit the transaction
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
            insert_data(conn, st.session_state['data_name'], st.session_state["uploaded_file"])
            st.success("Data saved into database")
        else:
            st.write("")
    except Exception as e:
        print(e)
        st.markdown("<span class='info-box'>Please upload any data</span>",
                    unsafe_allow_html=True)

    st.write("")