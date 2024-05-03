import mysql.connector
from mysql.connector import Error

def connect_to_starrocks(host, port, username, password):
    try:
        # Create a connection
        connection = mysql.connector.connect(
            host=host,
            port=port,
            user=username,
            password=password
        )
        if connection.is_connected():
            db_info = connection.get_server_info()
            print("Successfully connected to StarRocks server version ", db_info)
            # You can add your query execution here
            # cursor = connection.cursor()
            # cursor.execute("SELECT * FROM your_table;")
            # records = cursor.fetchall()
            # print("Total number of rows in table: ", cursor.rowcount)

            # cursor.close()
            connection.close()
            print("MySQL connection is closed")

    except Error as e:
        print("Error while connecting to StarRocks", e)

# Replace 'your_host', 'your_port', 'your_username', and 'your_password' with your StarRocks database credentials
# connect_to_starrocks(config['db_host'], '9030', config['db_user'], config['db_password'])
