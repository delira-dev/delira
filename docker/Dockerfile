FROM nvidia/cuda:9.2-base-ubuntu18.04

RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    build-essential \
    python3.6 \
    python3.6-dev \
    python3-pip \
    fonts-roboto \
    && rm -rf /var/lib/apt/lists/*
RUN rm -f /usr/bin/python && ln -s /usr/bin/python3 /usr/bin/python
RUN rm -f /usr/bin/pip && ln -s /usr/bin/pip3 /usr/bin/pip

RUN useradd --create-home --shell /bin/bash containeruser
USER containeruser
WORKDIR /home/containeruser

# install specific pip packages
RUN git clone https://gitlab+deploy-token-1:Gv_M5sEpMtPAziBsSVNz@git.lfb.rwth-aachen.de/Radiology/Delira.git && \
    pip install -r Delira/requirements_unix.txt
ENV PYTHONPATH /home/containeruser/Delira:$PYTHONPATH
CMD ["/bin/bash"]
