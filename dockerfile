FROM python:3.13-slim

# Install necessary packages
RUN apt-get update && apt-get install -y \
        build-essential \
        cmake \
        wget \
        git \
        docker.io

WORKDIR /app


COPY requirements.txt .

COPY . .

CMD ["/bin/bash"]