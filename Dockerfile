FROM python:3.6-slim-stretch

RUN apt-get -y update && apt-get install -y  \
    build-essential \
    cmake \
    gfortran \
    git \
    graphicsmagick \
    libgraphicsmagick1-dev \
    libatlas-dev \
    libavcodec-dev \
    libavformat-dev \
    libgtk2.0-dev \
    libjpeg-dev \
    liblapack-dev \
    libswscale-dev \
    pkg-config \
    python3-dev \
    python3-numpy \
    software-properties-common \
    zip \
    && apt-get clean && rm -rf /tmp/* /var/tmp/*

RUN  pip install opencv-python==3.4.1.15 \
        dlib==19.13.1 \
        numpy==1.14.5 \
        requests==2.19.0 \
        Pillow==5.1.0 \
        flask==1.0.2 \
        keras==2.2.0 \
        tensorflow==1.8.0 \
        flask-cors

COPY . /root/face_recognotion
WORKDIR /root/face_recognotion

EXPOSE 8080

CMD ["python3", "app.py"]

# docker build -t face_recognition:latest .
# docker run -it -p 8080:8080 face_recognition:latest
