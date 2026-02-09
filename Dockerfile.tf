# Base image: Use an official Python image with TensorFlow pre-installed
FROM tensorflow/tensorflow:2.20.0

# Set the working directory inside the container
WORKDIR /app

# Ensure pip is up to date
RUN apt-get install build-essential -y
RUN pip install --upgrade pip && \
    pip install pandas scipy pytest

COPY instmodel /app/instmodel
COPY MANIFEST.in setup.py setup.cfg /app/

RUN pip install --no-deps .

COPY tests/ tests/

# Default command to run when the container starts (can be changed)
CMD ["sh", "-c", "pytest tests/test_tf.py -v"]
