FROM nvidia/cuda:10.2-base-ubuntu16.04

RUN apt-get update && apt-get -y install xvfb ffmpeg build-essential git gcc

RUN apt-get -qq update && apt-get -qq -y install curl bzip2 \
    && curl -sSL https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -bfp /usr/local \
    && rm -rf /tmp/miniconda.sh \
    && conda install -y python=3.6 \
    && conda update conda \
    && apt-get -qq -y remove curl bzip2 \
    && apt-get -qq -y autoremove \
    && apt-get autoclean \
    && rm -rf /var/lib/apt/lists/* /var/log/dpkg.log \
    && conda clean --all --yes

ENV PATH /opt/conda/bin:$PATH

RUN pip install --upgrade pip

RUN git clone https://github.com/udacity/deep-reinforcement-learning.git
RUN cd deep-reinforcement-learning/python && pip install . \
&& pip install torch==1.6 torchvision --upgrade-strategy only-if-needed \
&& pip install tensorflow==2.3 tensorboard==2.3 grpcio==1.24.3

RUN pip install wandb --upgrade-strategy only-if-needed


