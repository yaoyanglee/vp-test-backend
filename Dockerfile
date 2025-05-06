# FROM python:3.12

# WORKDIR /code

# RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 

# COPY ./requirements.txt /code/requirements.txt

# RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# COPY ./app /code/app

# # Changed from 80 to 443
# EXPOSE "443"

# # CMD ["fastapi", "run", "app/app.py", "--port", "80"]
# CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "443"]

# Use Python base image
FROM python:3.12

# Set working directory
WORKDIR /code

# Install required system dependencies
RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 nginx openssl

# Copy and install dependencies
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy application files
COPY ./app /code/app

# # Copy the NGINX configuration file (you will create this next)
# COPY nginx.conf /etc/nginx/nginx.conf

# # Create self-signed SSL certificate (for testing)
# RUN mkdir -p /etc/nginx/ssl && \
#     openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
#     -keyout /etc/nginx/ssl/nginx.key \
#     -out /etc/nginx/ssl/nginx.crt \
#     -subj "/CN=localhost"

# Expose HTTPS and HTTP ports
EXPOSE 443 80

# Start NGINX and FastAPI
# CMD service nginx start && uvicorn app.app:app --host 0.0.0.0 --port 8000
CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8000"]
