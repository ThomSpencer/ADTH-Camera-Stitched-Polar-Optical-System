FROM python:3.13-slim

# Install necessary packages
RUN apt-get update && apt-get install -y --no-install-recommends \
                build-essential \
                cmake \
                wget \
                git \
                docker.io \
                fontconfig \
                fonts-dejavu-core \
                libgtk-3-0 \
                libgtk-3-dev \
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
                pkg-config \
                v4l-utils \
        && rm -rf /var/lib/apt/lists/*

ENV QT_QPA_FONTDIR=/usr/share/fonts/truetype/dejavu

WORKDIR /app


COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY . .

CMD ["/bin/bash"]