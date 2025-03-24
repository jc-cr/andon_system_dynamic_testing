FROM ubuntu:20.04 AS base

# Set noninteractive installation to avoid prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install Python and basic dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    apt-transport-https \
    apt-utils \
    curl \
    lsb-release \
    gnupg \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for Python if they don't exist
RUN if [ ! -e /usr/bin/python ]; then ln -s /usr/bin/python3 /usr/bin/python; fi && \
    if [ ! -e /usr/bin/pip ]; then ln -s /usr/bin/pip3 /usr/bin/pip; fi

# Second stage: Install RealSense SDK
FROM base AS realsense

# Install RealSense dependencies
RUN apt-get update && apt-get install -y \
    udev \
    libssl-dev \
    libusb-1.0-0-dev \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    libglfw3-dev \
    && rm -rf /var/lib/apt/lists/*

# Add the RealSense repository
RUN mkdir -p /etc/apt/keyrings && \
    curl -sSf https://librealsense.intel.com/Debian/librealsense.pgp | tee /etc/apt/keyrings/librealsense.pgp > /dev/null && \
    echo "deb [signed-by=/etc/apt/keyrings/librealsense.pgp] https://librealsense.intel.com/Debian/apt-repo focal main" | \
    tee /etc/apt/sources.list.d/librealsense.list

# Update and install RealSense packages
RUN apt-get update && apt-get install -y \
    librealsense2-dkms \
    librealsense2-utils \
    librealsense2-dev \
    librealsense2-dbg \
    && rm -rf /var/lib/apt/lists/*

# Final stage: Add Python libraries and finalize
FROM realsense AS python

# Install Python wrapper for RealSense
RUN pip install --no-cache-dir pyrealsense2 numpy opencv-python

# Set environment variables to fix GL issues
ENV DISPLAY=:0
ENV LIBGL_ALWAYS_SOFTWARE=1
ENV MESA_GL_VERSION_OVERRIDE=3.3
ENV __GLX_VENDOR_LIBRARY_NAME=mesa
ENV LIBGL_ALWAYS_INDIRECT=0
ENV QT_X11_NO_MITSHM=1



FROM python AS edgetpu

WORKDIR /home

RUN apt-get update && apt-get install -y \
    git \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# Add Edge TPU repository
RUN echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | tee /etc/apt/sources.list.d/coral-edgetpu.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -

# Install Edge TPU libraries
RUN apt-get update && apt-get install -y \
    libedgetpu1-std \
    python3-pycoral \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/lib/libedgetpu.so.1 /usr/lib/libedgetpu.so

RUN mkdir coral
WORKDIR /home/coral
RUN git clone https://github.com/google-coral/pycoral.git

WORKDIR /home/coral/pycoral
RUN bash examples/install_requirements.sh classify_image.py


FROM edgetpu AS example_object_detection


# Set working directory back to app
WORKDIR /home


RUN apt-get update && apt-get install -y \
    wget \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    python3-gi \
    python3-gst-1.0 \
    python3-numpy \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libgirepository1.0-dev \
    libcairo2-dev \
    && rm -rf /var/lib/apt/lists/*

ENV LD_LIBRARY_PATH=/usr/lib:/usr/local/lib:${LD_LIBRARY_PATH}


RUN git clone https://github.com/jc-cr/example-object-tracker.git
WORKDIR /home/example-object-tracker
RUN sh download_models.sh
WORKDIR gstreamer
RUN bash install_requirements.sh

RUN pip install --no-cache-dir matplotlib scikit-image filterpy


FROM example_object_detection AS final


# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-tk \
    libgl1-mesa-glx \
    libglib2.0-0 \
    usbutils \
    libqt5gui5 \
    libqt5core5a \
    libqt5widgets5 \
    libxcb-xinerama0 \
    libxkbcommon-x11-0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-xfixes0 \
    x11-utils \
    && rm -rf /var/lib/apt/lists/*


# Set QT platform and disable GUI threading warnings
ENV QT_X11_NO_MITSHM=1
ENV QT_PLUGIN_PATH=/usr/lib/x86_64-linux-gnu/qt5/plugins
ENV QT_DEBUG_PLUGINS=1
ENV PYTHONUNBUFFERED=1
ENV OPENCV_VIDEOIO_PRIORITY_MSMF=0

# Set environment variable for OpenCV GUI
ENV DISPLAY=:0


RUN pip install --no-cache-dir pillow requests eel


WORKDIR /app

# Default command
CMD ["bash"]

