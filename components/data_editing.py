from util import data_uploader_components, get_data

def data_editing_page(st):
    st.markdown("<h2 class='menu-title'>Data Editing</h2>",
                unsafe_allow_html=True)
    st.markdown("<h6 class='menu-subtitle'>Reviewing, cleaning, and modifying the dataset to address various data quality issues before using it to train a machine learning model</h6>",
                unsafe_allow_html=True)
    st.markdown("<hr class='menu-divider' />",
                unsafe_allow_html=True)

    # Bringing back the data from uploaded_file session_state
    if "uploaded_file" in st.session_state:

        # Assigning uploaded_file in session state to a variable
        dataframe = st.session_state.uploaded_file

        # Initiating data on session state
        if "data" not in st.session_state:
            st.session_state.data = dataframe

        st.markdown("<h3 class='menu-secondary'>Original Data</h3>",
                    unsafe_allow_html=True)
        st.write(dataframe)
        st.write(":green[Data Shape :]", dataframe.shape)

    else:
        data_uploader_components(st)
        if "uploaded_file" in st.session_state:
            st.write(st.session_state['data'])
            st.write(":green[Data Shape :]", st.session_state['data'].shape)

    if "uploaded_file" not in st.session_state:
        st.markdown("<span class='info-box'>Please upload any data</span>",
                    unsafe_allow_html=True)

    else:

        if st.checkbox('Edit Data'):

            # Initiating data on session state
            if "data" not in st.session_state:
                st.session_state.data = dataframe

            # Callback function to delete records in data
            def callback():
                edited_rows = st.session_state["data_editor"]["edited_rows"]
                rows_to_delete = []

                for idx, value in edited_rows.items():
                    if value["x"] is True:
                        rows_to_delete.append(idx)

                st.session_state["data"] = (
                    st.session_state["data"].drop(
                        rows_to_delete, axis=0).reset_index(drop=True)
                )

            # Configuring column to delete
            columns = st.session_state["data"].columns
            column_config = {column: st.column_config.Column(
                disabled=True) for column in columns}
            modified_df = st.session_state["data"].copy()
            modified_df["x"] = False

            # Moving delete column to be the first
            modified_df = modified_df[["x"] +
                                      modified_df.columns[:-1].tolist()]

            # Adding one space
            st.markdown("<br>", unsafe_allow_html=True)

            st.write("Please click the data on x to delete the record.")

            # Initating Data Editor
            edited_data = st.data_editor(
                modified_df,
                key="data_editor",
                on_change=callback,
                hide_index=False,
                column_config=column_config,
            )
            st.write(":green[Edited Data Shape, :]",
                     edited_data.drop(columns=['x'], axis=1).shape)

            if "edited_data" not in st.session_state:
                st.session_state["edited_data"] = edited_data.drop(columns=[
                                                                   'x'], axis=1)
            else:
                st.session_state["edited_data"] = edited_data.drop(columns=[
                                                                   'x'], axis=1)

        else:
            st.write("")