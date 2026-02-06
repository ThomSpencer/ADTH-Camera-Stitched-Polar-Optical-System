FROM python:3.13-slim

# Install necessary packages
RUN apt-get update && apt-get install -y --no-install-recommends \
                build-essential \
                cmake \
                wget \
                git \
                docker.io \
                libgl1 \
                libglib2.0-0 \
                libx11-6 \
                libxext6 \
                libxrender1 \
                libxfixes3 \
                libxkbcommon0 \
                libxkbcommon-x11-0 \
                libsm6 \
                libice6 \
                libxcb1 \
                libxrandr2 \
                libxi6 \
                libxinerama1 \
                v4l-utils \
        && rm -rf /var/lib/apt/lists/*

WORKDIR /app


COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["/bin/bash"]