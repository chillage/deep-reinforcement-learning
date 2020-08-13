FROM nvidia/cuda:10.2-base-ubuntu16.04
#FROM pytorch/pytorch:0.4_cuda9_cudnn7
#FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime

RUN apt-get update && \
    apt-get -y install xvfb ffmpeg git build-essential

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

RUN pip install gym[box2d,atari] pybullet==2.8.1
RUN git clone https://github.com/udacity/deep-reinforcement-learning.git
RUN cd deep-reinforcement-learning/python && pip install .



