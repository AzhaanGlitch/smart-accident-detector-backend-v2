# Use an official Python 3.10 runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container to /app
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application's code (app.py, .h5 models, etc.)
COPY . .

# Make port 7860 available to the platform
EXPOSE 7860

# Command to run the application using a production-ready Gunicorn server.
# --preload loads the ML models once before the server starts, preventing timeouts.
CMD ["gunicorn", "app:app", "--workers=1", "--bind=0.0.0.0:7860", "--timeout=300", "--preload"]