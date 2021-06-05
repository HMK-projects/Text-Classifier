# This dockerfile is used for development.

# Use TF 2.5 as base image with GPU support
tensorflow/tensorflow:2.4.1-gpu

# Set working directory to text-classifier
WORKDIR /text-classifier
# Add the requirements.txt to the working directory
ADD ./requirements.txt .

# View contents while building dockerfile
RUN ls -a

# Install dependencies
RUN pip install -r requirements.txt