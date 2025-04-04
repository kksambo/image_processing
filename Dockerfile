# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container
COPY . /app

# Install required Python packages
RUN pip install --no-cache-dir flask flask-cors tensorflow pillow

# Suppress TensorFlow logs
ENV TF_CPP_MIN_LOG_LEVEL=2

# Expose the port Render will use
EXPOSE 5000

# Command to run the Flask app
CMD ["python", "app.py"]
