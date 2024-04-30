# Use Python 3.8 image from Docker Hub as the base image
FROM python:3.8

# Set the environment variable for the timezone
ENV TZ=Asia/Bangkok

# Install tzdata package and configure the timezone
RUN apt-get update && apt-get install -y tzdata \
    && ln -fs /usr/share/zoneinfo/$TZ /etc/localtime \
    && dpkg-reconfigure -f noninteractive tzdata

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file into the container at /app
COPY requirements.txt /app/

# Install dependencies from the requirements.txt file
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application from the local context to /app in the container
COPY . /app

# Command to run on container start
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=7777"]
