from util import data_uploader_components, get_data
from streamlit_pandas_profiling import st_profile_report
import psycopg2
from util import config
import json
from datetime import datetime

def insert_data(conn, dataset_name, dataframe):
    # Convert DataFrames to JSON strings as necessary
    dataframe_json = json.dumps(dataframe.to_dict(orient='records'))

    # Generate data correlation matrix and convert to JSON
    data_correlation = json.dumps(dataframe.corr().to_dict())
    
    # Generate data description (summary statistics)
    data_description = json.dumps(dataframe.describe().to_dict())
    
    # Count variables (columns) and observations (rows)
    variable_number = len(dataframe.columns)
    observation_number = len(dataframe)
    
    # Establish a connection to the database
    cursor = conn.cursor()

    # Prepare and execute the insertion query
    cursor.execute(
        """
        INSERT INTO data_exploration (
            dataset, dataframe, data_correlation, variable_number, 
            observation_number, data_description, created_at
        ) VALUES (%s, %s, %s, %s, %s, %s, %s)
        """,
        (dataset_name, dataframe_json, data_correlation, 
         variable_number, observation_number, data_description, datetime.now())
    )
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
    CREATE TABLE IF NOT EXISTS data_exploration (
        id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
        dataset VARCHAR(50),
        dataframe JSON,
        data_correlation JSON,
        variable_number INT,
        observation_number INT,
        data_description JSON,
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