# Use an official Python base image with Python 3.12
FROM python:3.12-slim

WORKDIR /opt/app

# Set environment variables to prevent Python from buffering stdout and stdin
ENV PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y libpq-dev python3-dev gcc \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --upgrade pip

COPY app app
COPY certs certs
COPY requirements.txt .
COPY .env .

RUN mkdir -p logs
RUN mkdir -p models

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt gunicorn

RUN rm requirements.txt

# Make port available to the world outside this container
EXPOSE 443

# Run the application - path is /opt/app/app/app_main.py
CMD ["gunicorn", "--certfile=certs/app_certificate.pem", "--keyfile=certs/app_private_key.pem", "--log-level=debug", "--workers=4", "--threads=2", "--bind=0.0.0.0:443", "app.app_main:app"]
