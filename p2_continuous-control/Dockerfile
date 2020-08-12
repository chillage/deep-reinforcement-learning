FROM pytorch/pytorch:0.4_cuda9_cudnn7

RUN apt-get update && \
    apt-get -y install xvfb ffmpeg git build-essential
RUN pip install --upgrade pip
RUN pip install gym[box2d,atari] pybullet==2.8.1
RUN git clone https://github.com/vwxyzjn/cleanrl && \
    cd cleanrl && pip install -e .
RUN apt-get -y install python-opengl
RUN pip install opencv-python
RUN pip install seaborn pandas
RUN pip install stable-baselines3
RUN pip install wandb
RUN wandb login 3c5ceb4355fe069912a8ccad904c3ce950330282
WORKDIR /workspace/cleanrl/cleanrl

COPY entrypoint.sh /usr/local/bin/
RUN chmod 777 /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD python ppo2_continuous_action.py --capture-video --total-timesteps 200
